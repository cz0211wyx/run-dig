import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from runshelf.cli import main
from runshelf.scan import analyze
from runshelf.report import write_report


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, path, value):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            file.write_bytes(value)
        else:
            file.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return file

    def run_folder(self, name, **config):
        self.put(name + "/config.json", {"train": {"lr": .001, "epochs": 10, "batch_size": 16, "model": "tiny", **config}})
        self.put(name + "/best.pt", name.encode())
        self.put(name + "/history.jsonl", '{"epoch":1,"loss":2}\n{"epoch":2,"loss":1}\n')

    def report(self, **kwargs):
        return analyze(self.root, **kwargs)

    def test_existing_listing_sort_and_filter(self):
        self.put("a/best.PTH", b"12")
        self.put("b/best.pt", b"12345")
        self.put("b/last.pt", b"1")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["list", str(self.root), "--sort", "size", "--name", "BEST"]), 0)
        lines = output.getvalue().splitlines()
        self.assertIn("Found 2 checkpoints", lines[0])
        self.assertTrue(lines[3].startswith("best.pt"))

    def test_loose_arbitrary_names_remain_uncertain(self):
        self.put("new folder/final_final2.pt", b"model-a")
        self.put("new folder/final.pt", b"model-b")
        self.put("random name/checkpoint.ckpt", b"model-c")
        report = self.report()
        self.assertEqual(len(report.experiments), 3)
        self.assertTrue(all(n.kind == "checkpoint_candidate" and n.confidence < .5 for n in report.experiments))
        self.assertEqual(report.edges, [])

    def test_checkpoint_subdirectory_belongs_to_run(self):
        self.put("whatever/config.yaml", "seed: 42\nlr: 0.1\nepochs: 3\n")
        self.put("whatever/checkpoints/epoch-9.ckpt", b"opaque")
        self.assertEqual([(n.path, n.checkpoint_count) for n in self.report().experiments], [("whatever", 1)])

    def test_config_subdirectory_layout(self):
        self.put("new folder/.hydra/config.yaml", "seed: 42\nlr: 0.1\nepochs: 3\n")
        self.put("new folder/weights/model.pt", b"opaque")
        report = self.report()
        self.assertEqual(len(report.experiments), 1)
        self.assertEqual(report.experiments[0].config["seed"], 42)

    def test_nested_run_not_swallowed_by_parent(self):
        self.run_folder("parent")
        self.run_folder("parent/nested")
        self.assertEqual([n.checkpoint_count for n in self.report().experiments], [1, 1])

    def test_logs_and_checkpoints_without_config(self):
        self.put('messy folder/final.pt', b'one')
        self.put('messy folder/final_final2.pt', b'two')
        self.put('messy folder/train.log', 'epoch=1 loss=0.8\nepoch=2 loss=0.4')
        report = self.report()
        self.assertEqual(len(report.experiments), 1)
        self.assertEqual(report.experiments[0].checkpoint_count, 2)
        self.assertEqual(report.experiments[0].confidence, .7)

    def test_arbitrary_config_filename(self):
        self.put('scratch/whatever.json', {'lr': .01, 'batch_size': 32, 'seed': 1})
        self.put('scratch/final2.pt', b'opaque')
        self.assertEqual(self.report().experiments[0].config['batch_size'], 32)

    def test_explicit_resume_edge_and_evidence(self):
        self.run_folder("base")
        self.run_folder("new folder", resume_from="../base/best.pt")
        report = self.report()
        relations = [e for e in report.edges if e.kind == "derived_from"]
        self.assertEqual(len(relations), 1)
        by_id = {n.id: n.path for n in report.experiments}
        self.assertEqual((by_id[relations[0].source], by_id[relations[0].target]), ("base", "new folder"))
        self.assertIn("config.json", relations[0].evidence[0]["artifact"])

    def test_initialization_and_component_paths_need_no_hashes(self):
        for name in ('new folder', 'another model', 'consumer'):
            self.run_folder(name)
        self.put('consumer/config.json', {'lr': .01, 'batch_size': 16,
                                         'initial_backbone_source': '../new folder/best.pt',
                                         'experts': ['../another model/best.pt']})
        report = self.report(hash_limit=0)
        names = {node.id: node.path for node in report.experiments}
        found = {(names[e.source], names[e.target], e.kind) for e in report.edges if e.directed}
        self.assertEqual(found, {('new folder', 'consumer', 'derived_from'), ('another model', 'consumer', 'references')})

    def test_seed_path_template_uses_recorded_value(self):
        self.run_folder('source_17')
        self.run_folder('target')
        self.put('target/config.json', {'seed': 17, 'lr': .01, 'batch_size': 16,
                                        'joint_initialization': '../source_{seed}/best.pt'})
        report = self.report(hash_limit=0)
        self.assertEqual([e.kind for e in report.edges], ['derived_from'])
        self.assertIn('source_17/best.pt', report.edges[0].evidence[0]['matched_artifacts'])

    def test_ambiguous_template_is_not_guessed(self):
        self.run_folder('source_17')
        self.run_folder('target', init_from='../source_{seed}/best.pt')
        self.put('target/config.json', {'lr': .01, 'batch_size': 16, 'a.seed': 17, 'b.seed': 99,
                                        'init_from': '../source_{seed}/best.pt'})
        report = self.report()
        self.assertFalse(any(e.directed for e in report.edges))
        self.assertEqual(report.statistics['unresolved_references'], 1)

    def test_manifest_stage_dependencies_need_no_hashes(self):
        self.run_folder('renamed/source')
        self.run_folder('random final')
        self.put('pipeline.json', {'stages': [
            {'name': 'encoder', 'checkpoint': 'renamed/source/best.pt'},
            {'name': 'consumer', 'checkpoint': 'random final/best.pt', 'components': ['encoder']}]})
        report = self.report(hash_limit=0)
        names = {node.id: node.path for node in report.experiments}
        relations = [e for e in report.edges if e.directed]
        self.assertEqual([(names[e.source], names[e.target]) for e in relations], [('renamed/source', 'random final')])
        self.assertEqual(relations[0].evidence[0]['artifact'], 'pipeline.json')

    def test_colocated_runs_without_dependencies_are_not_connected_as_sources(self):
        for name in ('a', 'b', 'c'):
            self.run_folder('collection/' + name)
        self.put('collection/state.json', {'completed': [{'name': name, 'checkpoint': name + '/best.pt'} for name in ('a', 'b', 'c')]})
        self.assertFalse(any(edge.directed for edge in self.report().edges))

    def test_manifest_frozen_components_bind_renamed_runs(self):
        import hashlib
        self.run_folder('arbitrary source')
        self.run_folder('arbitrary target')
        fingerprint = lambda text: hashlib.sha256(text.encode()).hexdigest()
        self.put('batch/state.json', {'completed': [
            {'phase': 'branch red', 'checkpoint_sha256': fingerprint('arbitrary source')},
            {'phase': 'combined', 'checkpoint_sha256': fingerprint('arbitrary target'),
             'frozen_components_unchanged': True,
             'frozen_component_state_sha256': {'branch red': {str(i): fingerprint(str(i)) for i in range(4)}}}]})
        report = self.report()
        relations = [e for e in report.edges if e.directed]
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].kind, 'references')
        self.assertLess(relations[0].confidence, .9)
        self.assertTrue(relations[0].caveats)
        # Duplicate bytes cannot identify which run supplied the component.
        self.run_folder('duplicate')
        self.put('duplicate/best.pt', b'arbitrary source')
        ambiguous = self.report()
        self.assertFalse(any(e.directed for e in ambiguous.edges))
        self.assertTrue(any('multiple runs' in w['message'] for w in ambiguous.warnings))

    def test_evaluation_with_config_and_metrics_is_a_graph_node(self):
        self.run_folder('trained')
        self.put('evaluation/inference_config.json', {'checkpoint': '../trained/best.pt'})
        self.put('evaluation/metrics.json', {'accuracy': .9})
        report = self.report(hash_limit=0)
        names = {node.id: node.path for node in report.experiments}
        self.assertTrue(any(names[e.source] == 'trained' and names[e.target] == 'evaluation' for e in report.edges))

    def test_shared_parent_config_is_tentative_and_never_overrides_local_config(self):
        self.run_folder('source')
        self.put('trial/config.json', {'lr': .01, 'batch_size': 16, 'initial_model': '../source/best.pt'})
        self.put('trial/a/best.pt', b'a')
        self.put('trial/a/train.log', 'epoch=1 loss=0.5')
        self.run_folder('trial/b')
        report = self.report(hash_limit=0)
        names = {node.id: node.path for node in report.experiments}
        inferred = [edge for edge in report.edges if edge.directed]
        self.assertEqual([(names[e.source], names[e.target], e.kind) for e in inferred], [('source', 'trial/a', 'possible_derivation')])
        self.assertLessEqual(inferred[0].confidence, .65)
        self.assertTrue(any('Shared parent' in c for c in inferred[0].caveats))

    def test_log_quoted_path_and_static_script_path(self):
        self.run_folder('old folder')
        self.run_folder('log consumer')
        self.run_folder('script consumer')
        self.put('log consumer/train.log', 'Resuming from "../old folder/best.pt"\n')
        self.put('script consumer/train.py', 'from pathlib import Path\nbase = Path("../old folder")\ncheckpoint = base / "best.pt"\nmodel = load_checkpoint(checkpoint)\nraise RuntimeError("do not execute")\n')
        report = self.report(hash_limit=0)
        names = {node.id: node.path for node in report.experiments}
        relations = {names[e.target]: e for e in report.edges if e.directed}
        self.assertEqual(set(relations), {'log consumer', 'script consumer'})
        self.assertEqual(relations['log consumer'].kind, 'derived_from')
        self.assertLessEqual(relations['script consumer'].confidence, .65)

    def test_inspect_checkpoint_prints_context_without_writing_report(self):
        self.run_folder('parent')
        self.run_folder('child', resume_from='../parent/best.pt')
        output = io.StringIO()
        before = sorted(p.relative_to(self.root) for p in self.root.rglob('*'))
        with contextlib.redirect_stdout(output):
            result = main(['inspect', 'child/best.pt', '--root', str(self.root)])
        self.assertEqual(result, 0)
        for text in ('Checkpoint:', 'Size:', 'Modified:', 'Experiment: child', 'Sources:', 'parent', 'Run configuration:', 'train.lr', 'Run metrics:'):
            self.assertIn(text, output.getvalue())
        self.assertEqual(before, sorted(p.relative_to(self.root) for p in self.root.rglob('*')))
        full = io.StringIO()
        with contextlib.redirect_stdout(full):
            self.assertEqual(main(['inspect', 'child/best.pt', '--root', str(self.root), '--full']), 0)
        self.assertIn('train.resume_from', full.getvalue())

    def test_conditional_script_paths_are_not_resolved_as_constants(self):
        self.run_folder('base')
        self.run_folder('target')
        self.put('target/train.py', 'checkpoint = "../base/best.pt"\nif something:\n    checkpoint = unknown_value\nload_checkpoint(checkpoint)\n')
        self.assertFalse(any(edge.directed for edge in self.report().edges))

    def test_inspect_supports_checkpoint_subfolder_and_missing_files(self):
        self.put('run/config.json', {'lr': .01, 'batch_size': 16})
        checkpoint = self.put('run/checkpoints/final.pt', b'opaque')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['inspect', str(checkpoint)]), 0)
        self.assertIn('lr: 0.01', output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['inspect', str(self.root / 'absent.pt')]), 2)
            self.assertEqual(main(['inspect', str(checkpoint), '--root', str(self.root / 'other')]), 2)

    def test_reference_is_not_automatically_derivation(self):
        self.run_folder("base")
        self.run_folder("evaluator", references=[{"checkpoint": "../base/best.pt"}])
        self.assertEqual([e.kind for e in self.report().edges], ["references"])

    def test_sha_reference_map(self):
        import hashlib
        self.run_folder("base")
        self.run_folder("target", source_checkpoints={"../base/best.pt": hashlib.sha256(b"base").hexdigest()})
        edge = self.report().edges[0]
        self.assertEqual(edge.kind, "references")
        self.assertGreater(edge.confidence, .95)

    def test_wrong_reference_hash_not_trusted(self):
        self.run_folder("base")
        self.run_folder("target", references=[{"checkpoint": "../base/best.pt", "sha256": "0" * 64}])
        report = self.report()
        self.assertFalse(any(e.kind == "references" for e in report.edges))
        self.assertEqual(report.statistics["unresolved_references"], 1)

    def test_duplicate_suffix_stays_ambiguous(self):
        self.run_folder("a/run")
        self.run_folder("b/run")
        self.run_folder("target", resume_from="/old/machine/run/best.pt")
        report = self.report()
        target = next(n for n in report.experiments if n.path == "target")
        self.assertEqual(target.references[0]["resolution"], "ambiguous")
        self.assertFalse(any(e.kind == "derived_from" for e in report.edges))

    def test_copy_identity_has_no_direction(self):
        self.put("final.pt", b"identical bytes")
        self.put("new folder/final_final2.pt", b"identical bytes")
        edge = self.report().edges[0]
        self.assertEqual(edge.kind, "same_artifact")
        self.assertFalse(edge.directed)
        self.assertEqual(edge.confidence, 1)

    def test_mtime_cannot_create_ancestry(self):
        self.run_folder("one")
        self.run_folder("two")
        os.utime(self.root / "one/best.pt", (100, 100))
        os.utime(self.root / "two/best.pt", (200, 200))
        report = self.report()
        self.assertTrue(any(e.kind == "similar_to" for e in report.edges))
        self.assertTrue(all(not e.directed for e in report.edges))

    def test_all_supported_similarities_are_retained(self):
        for name in ['a', 'b', 'c', 'd', 'e', 'f']:
            self.run_folder(name)
        report = self.report()
        similar = [edge for edge in report.edges if edge.kind == 'similar_to']
        self.assertEqual(len(similar), 15)
        first = report.experiments[0].id
        self.assertEqual(sum(first in (edge.source, edge.target) for edge in similar), 5)
        self.assertTrue(all(not edge.directed for edge in similar))

    def test_multi_generation_sources_are_retained(self):
        self.run_folder('base')
        self.run_folder('middle', resume_from='../base/best.pt')
        self.run_folder('leaf', resume_from='../middle/best.pt')
        self.run_folder('sibling', resume_from='../base/best.pt')
        report = self.report()
        names = {node.id: node.path for node in report.experiments}
        sources = {(names[edge.source], names[edge.target]) for edge in report.edges if edge.kind == 'derived_from'}
        self.assertEqual(sources, {('base', 'middle'), ('middle', 'leaf'), ('base', 'sibling')})

    def test_strong_similarity_and_recorded_time_is_only_tentative(self):
        options = {"dropout": .1, "layers": 2, "hidden_size": 32, "optimizer": "adam"}
        self.run_folder("one", **options)
        self.run_folder("two", **options, lr=.002)
        self.put("one/source/model.py", "MODEL = 'same'\n")
        self.put("two/source/model.py", "MODEL = 'same'\n")
        self.put("one/state.json", {"started_at_utc": "2026-01-01T00:00:00Z", "finished_at_utc": "2026-01-02T00:00:00Z"})
        self.put("two/state.json", {"started_at_utc": "2026-01-03T00:00:00Z"})
        relation = self.report().edges[0]
        self.assertEqual(relation.kind, "possible_derivation")
        self.assertLessEqual(relation.confidence, .75)

    def test_sequential_seed_repeats_are_not_inferred_ancestry(self):
        for name, seed, date in [('one', 1, '2026-01-01'), ('two', 2, '2026-01-02')]:
            self.run_folder(name, seed=seed)
            self.put(name + '/source/train.py', '# identical code\n')
            self.put(name + '/state.json', {'started_at_utc': date + 'T00:00:00Z', 'finished_at_utc': date + 'T01:00:00Z'})
        self.assertTrue(all(not e.directed for e in self.report().edges))

    def test_launcher_directory_is_not_one_large_experiment(self):
        self.run_folder('runs/one')
        self.run_folder('runs/two')
        self.put('runs/one.status.json', {'started_at_utc': '2026-01-01T00:00:00Z'})
        self.put('runs/two.log', 'epoch=1 loss=0.5')
        self.assertNotIn('runs', [n.path for n in self.report().experiments])

    def test_cycles_are_flagged(self):
        self.run_folder("one", resume_from="../two/best.pt")
        self.run_folder("two", resume_from="../one/best.pt")
        edges = [e for e in self.report().edges if e.directed]
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(any("cycle" in c for c in e.caveats) for e in edges))

    def test_invalid_json_retains_checkpoint_and_warning(self):
        self.put("folder/config.json", "{broken")
        self.put("folder/best.pt", b"opaque")
        report = self.report()
        self.assertEqual(report.statistics["checkpoints"], 1)
        self.assertTrue(report.warnings)

    def test_large_text_is_bounded_and_reported(self):
        self.put("config.json", '{"value":"' + "x" * 200 + '"}')
        report = self.report(text_limit=30)
        self.assertTrue(any("exceeds" in w["message"] for w in report.warnings))
        self.assertEqual(report.artifacts[0].metadata, {})

    def test_csv_and_jsonl_metric_extraction(self):
        self.run_folder("run")
        self.put("run/metrics.csv", "epoch,accuracy\n1,0.5\n2,0.8\n")
        self.put("run/history.jsonl", '{"loss":2}\nbroken\n{"loss":1}\n')
        n = self.report().experiments[0]
        self.assertEqual(n.metrics["metrics.csv:accuracy"], .8)
        self.assertEqual(n.metrics["history.jsonl:loss"], 1)
        self.assertTrue(n.warnings)

    def test_scripts_are_never_executed(self):
        self.run_folder("run")
        self.put("run/source/train.py", "raise RuntimeError('must not execute')\n")
        self.assertEqual(self.report().statistics["checkpoints"], 1)

    def test_shell_reference_is_static_and_capped(self):
        self.run_folder('base')
        self.run_folder('next')
        self.put('next/train.sh', 'python train.py --resume-from "../base/best.pt" --seed 42')
        edges = [e for e in self.report().edges if e.kind == 'derived_from']
        self.assertEqual(len(edges), 1)
        self.assertLessEqual(edges[0].confidence, .65)
        self.assertTrue(any('Static' in c for c in edges[0].caveats))

    def test_redaction(self):
        self.run_folder("run", api_key="do-not-leak", credentials={"password": "hide-me"})
        payload = json.dumps(self.report().to_dict())
        self.assertNotIn("do-not-leak", payload)
        self.assertNotIn("hide-me", payload)

    def test_output_is_offline_and_escapes_html(self):
        self.run_folder("run", model="</script><script>alert(1)</script>")
        output = self.root / "report"
        page = write_report(self.report(output=output), output)
        html = page.read_text(encoding="utf-8")
        self.assertNotIn("</script><script>alert(1)", html)
        self.assertIn("\\u003c/script", html)
        self.assertNotIn("<script src=", html)
        parsed = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(len(self.report(output=output).experiments), 1)

    def test_output_will_not_overwrite_unrelated_directory(self):
        output = self.root / "unrelated"
        self.put("unrelated/index.html", "keep me")
        with self.assertRaises(ValueError):
            write_report(self.report(), output)
        self.assertEqual((output / "index.html").read_text(), "keep me")

    def test_hash_budget_and_excludes(self):
        self.put("a.pt", b"1234")
        self.put("b.pt", b"1234")
        self.put("node_modules/hidden.pt", b"1")
        self.put("backup/c.pt", b"1")
        report = self.report(hash_budget=4, excludes=["backup"])
        self.assertEqual(report.statistics["checkpoints"], 2)
        self.assertEqual(report.statistics["unhashed_checkpoints"], 1)
        self.assertFalse(report.edges)

    def test_partial_scan_is_visible(self):
        self.put("a.pt", b"1")
        self.put("b.pt", b"2")
        self.assertTrue(self.report(max_files=1).statistics["partial"])

    def test_symlink_cycle_is_not_followed(self):
        self.put('run/file.pt', b'bytes')
        try:
            (self.root / 'run/loop').symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest('Creating symlinks requires a platform privilege')
        self.assertEqual(self.report().statistics['checkpoints'], 1)

    def test_unreadable_subtree_reports_partial_scan(self):
        from unittest.mock import patch
        def failing_walk(root, followlinks=False, onerror=None):
            onerror(PermissionError(13, 'permission denied', str(root / 'private')))
            return iter([])
        with patch('runshelf.scan.os.walk', failing_walk):
            report = self.report()
        self.assertTrue(report.statistics['partial'])
        self.assertEqual(report.warnings[0]['kind'], 'read_error')

    def test_similarity_budget_reports_partial_inference(self):
        for name in ['a', 'b', 'c']:
            self.run_folder(name)
        report = self.report(max_pairs=1)
        self.assertTrue(report.statistics['partial'])
        self.assertTrue(any('Similarity candidate budget' in w['message'] for w in report.warnings))

    def test_config_conflicts_preserve_both_sources(self):
        self.run_folder('run')
        self.put('run/configs/override.json', {'lr': .2, 'batch_size': 10, 'train': {'lr': .5}})
        run = self.report().experiments[0]
        self.assertEqual(len(run.config_sources['train.lr']), 2)
        self.assertTrue(any('Conflicting config' in w for w in run.warnings))

    def test_ids_stable_across_rescan(self):
        self.run_folder("run")
        first, second = self.report(), self.report()
        self.assertEqual(first.experiments, second.experiments)
        self.assertEqual(first.edges, second.edges)

    def test_empty_tree(self):
        self.assertEqual(self.report().statistics["experiments"], 0)

    def test_cli_end_to_end(self):
        self.run_folder("run")
        output = self.root / "report"
        result = subprocess.run([sys.executable, "-m", "runshelf", "scan", str(self.root), "-o", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "index.html").is_file())
        self.assertIn("Found 1 checkpoints", result.stdout)


if __name__ == "__main__":
    unittest.main()
