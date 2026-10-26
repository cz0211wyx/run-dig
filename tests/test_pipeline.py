import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from rundig.cli import main
from rundig.scan import analyze
from rundig.report import write_report


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

    def npz(self, path, names=("W1", "b1", "epoch")):
        file = self.put(path, b"")
        with zipfile.ZipFile(file, "w") as archive:
            for name in names:
                # Recognition must not read array payloads, including object arrays.
                archive.writestr(name + ".npy", b"opaque array payload")
        return file

    def test_npz_parameters_recognized_and_feature_caches_excluded(self):
        self.npz("new folder/final_final2.NPZ")
        self.npz("another folder/arbitrary.npz", ("dense/kernel", "dense/bias"))
        self.npz("checkpoints/best.npz", ("features", "ids"))
        self.npz("cache/weights.npz", ("weights", "labels"))
        self.npz("checkpoints/predictions_by_epoch.npz", ("probabilities", "predicted_scores"))
        self.npz("reports/ig_full_final.npz", ("positions", "residual", "steps"))
        self.npz("data/unknown.npz", ("arr_0",))
        report = self.report(hash_limit=0)
        self.assertEqual({a.path for a in report.artifacts if a.kind == "checkpoint"},
                         {"new folder/final_final2.NPZ", "another folder/arbitrary.npz"})

    def test_npz_opaque_arrays_are_not_deserialized(self):
        self.npz("snapshot.npz", ("layer.weight", "layer.bias"))
        with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("Array payload read")):
            self.assertEqual(self.report().statistics["checkpoints"], 1)

    def test_npz_naming_fallback_and_broken_archives_keep_notes(self):
        self.npz("checkpoints/something.npz", ("arr_0",))
        broken = self.put("run/epoch_50.npz", b"PK\x03\x04truncated")
        self.put("unknown.npz", b"not a zip")
        report = self.report()
        self.assertEqual(report.statistics["checkpoints"], 2)
        self.assertTrue(all(a.warnings for a in report.artifacts if a.kind == "checkpoint"))
        self.assertTrue(any(w["path"] == "unknown.npz" for w in report.warnings))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["inspect", str(broken)]), 0)
        self.assertIn("Cannot read NPZ archive", output.getvalue())

    def test_npz_inspection_limit_is_reported(self):
        self.npz("run/best.npz")
        self.npz("unknown.npz")
        with patch("rundig.scan.NPZ_INSPECTION_LIMIT", 1):
            with patch("rundig.scan.zipfile.ZipFile", side_effect=AssertionError("Oversized archive opened")):
                report = self.report(hash_limit=0)
        self.assertEqual(report.statistics["checkpoints"], 1)
        self.assertEqual(len(report.warnings), 2)
        self.assertTrue(all("inspection skipped" in w["message"] for w in report.warnings))

    def test_npz_resume_and_evaluation_paths_are_resolved(self):
        self.npz("base/checkpoints/last.npz")
        self.put("base/config.json", {"lr": .1, "seed": 17})
        self.npz("resumed/checkpoints/best.npz")
        self.put("resumed/config.json", {"lr": .1, "seed": 17,
                                         "resume": r"base\checkpoints\last.npz"})
        self.put("resumed/eval.json", {"checkpoint": r"resumed\checkpoints\best.npz",
                                      "best_epoch": 14, "accuracy": .9, "macro_f1": .8})
        report = self.report(hash_limit=0)
        names = {e.id: e.path for e in report.experiments}
        self.assertEqual([(names[e.source], names[e.target]) for e in report.edges if e.kind == "derived_from"],
                         [("base", "resumed")])
        self.assertEqual(report.statistics["unresolved_references"], 0)
        records = self.checkpoint_records(report, "resumed/checkpoints/best.npz")
        self.assertEqual(records[0]["values"], {"best_epoch": 14, "accuracy": .9, "macro_f1": .8})
        self.assertEqual(records[0]["confidence"], .95)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["compare", "base/checkpoints/last.npz", "resumed/checkpoints/best.npz",
                                   "--root", str(self.root)]), 0)
        self.assertIn("A -> B: derived from", output.getvalue())
        self.assertIn("accuracy", output.getvalue())

    def test_npz_references_do_not_turn_data_into_models(self):
        self.npz("features/cache.npz", ("features", "ids"))
        self.npz("run/best.npz")
        self.put("run/config.json", {"lr": .1, "seed": 17, "resume": "features/cache.npz"})
        report = self.report()
        self.assertEqual(report.statistics["checkpoints"], 1)
        self.assertEqual(report.statistics["unresolved_references"], 1)
        self.assertFalse(any(e.directed for e in report.edges))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["inspect", "features/cache.npz", "--root", str(self.root)]), 2)

    def test_checkpoint_records_do_not_link_prediction_caches(self):
        self.npz("run/best.npz")
        self.npz("run/predictions_by_epoch.npz", ("probabilities", "predicted_scores"))
        self.put("consumer/config.json", {"lr": .1, "seed": 17, "checkpoints": [
            {"checkpoint": "run/best.npz", "prediction_cache": "run/predictions_by_epoch.npz"}]})
        self.put("consumer/best.pt", b"model")
        report = self.report(hash_limit=0)
        consumer = next(e for e in report.experiments if e.path == "consumer")
        self.assertEqual([r["value"] for r in consumer.references], ["run/best.npz"])
        self.assertEqual(report.statistics["unresolved_references"], 0)

    def test_npz_checkpoint_folder_and_metrics_establish_run_without_config(self):
        self.npz("missing config/checkpoints/best.npz")
        self.npz("missing config/checkpoints/last.npz")
        self.put("missing config/metrics.csv", "epoch,loss\n1,0.8\n2,0.4\n")
        self.put("missing config/eval.json", {"checkpoint": "checkpoints/best.npz", "accuracy": .9})
        report = self.report()
        self.assertEqual([(e.path, e.checkpoint_count) for e in report.experiments], [("missing config", 2)])
        self.assertEqual(self.checkpoint_records(report, "missing config/checkpoints/best.npz")[0]["values"],
                         {"accuracy": .9})

    def test_npz_references_from_logs_scripts_and_manifests(self):
        for name in ("base", "logged", "scripted", "stage"):
            self.npz(name + "/best.npz")
            self.put(name + "/config.json", {"lr": .1, "seed": 17})
        self.put("logged/train.log", 'Resuming from "../base/best.npz"')
        self.put("scripted/train.py", 'resume = "../base/best.npz"\nmodel = load_checkpoint(resume)')
        self.put("stages.json", {"stages": [
            {"name": "first", "checkpoint": "base/best.npz"},
            {"name": "second", "checkpoint": "stage/best.npz", "resume": "first"}]})
        report = self.report(hash_limit=0)
        names = {e.id: e.path for e in report.experiments}
        self.assertEqual({(names[e.source], names[e.target]) for e in report.edges if e.directed},
                         {("base", "logged"), ("base", "scripted"), ("base", "stage")})

    def test_npz_and_safetensors_in_terminal_listing(self):
        self.npz("run/best.npz")
        self.npz("features/best.npz", ("features", "ids"))
        self.put("run/best.safetensors", b"opaque")
        self.put("run/best.pt", b"opaque")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["list", str(self.root), "--name", "best"]), 0)
        self.assertIn("Found 3 checkpoints", output.getvalue())
        self.assertIn("best.npz", output.getvalue())
        self.assertNotIn("features", output.getvalue())

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

    def test_shared_project_config_does_not_describe_copied_child_checkpoint(self):
        self.run_folder('runs/one')
        self.put('configs/base.json', {'lr': .2, 'epochs': 40, 'batch_size': 32})
        self.put('checkpoints/best.pt', b'runs/one')
        report = self.report()
        nodes = {node.path: node for node in report.experiments}
        self.assertNotIn('.', nodes)
        copied = nodes['checkpoints/best.pt']
        self.assertEqual(copied.kind, 'checkpoint_candidate')
        self.assertEqual(copied.config, {})
        self.assertTrue(any('was not bound' in note for note in copied.warnings))
        relations = [edge for edge in report.edges if copied.id in (edge.source, edge.target)]
        self.assertEqual([edge.kind for edge in relations], ['same_artifact'])
        self.assertFalse(relations[0].directed)
        # Missing hashes cannot establish that these are copies.
        self.assertIn('.', [node.path for node in self.report(hash_limit=0).experiments])

    def test_support_config_runs_keep_independent_training_evidence(self):
        for case in ('single', 'different', 'log', 'state', 'reference', 'hydra'):
            folder = '.hydra' if case == 'hydra' else 'configs'
            config = {'lr': .1, 'epochs': 3, 'batch_size': 16}
            if case == 'reference':
                config['checkpoint'] = 'checkpoints/best.pt'
            self.put(f'{case}/{folder}/config.json', config)
            if case != 'single':
                self.run_folder(f'{case}/runs/one')
            content = b'own weights' if case in {'single', 'different'} else f'{case}/runs/one'.encode()
            self.put(f'{case}/checkpoints/best.pt', content)
            if case == 'log':
                self.put(f'{case}/train.log', 'epoch=3 loss=0.2')
            if case == 'state':
                self.put(f'{case}/state.json', {'started_at_utc': '2026-01-01T00:00:00Z'})
        nodes = {node.path: node for node in self.report().experiments}
        for case in ('single', 'different', 'log', 'state', 'reference', 'hydra'):
            with self.subTest(case=case):
                self.assertEqual(nodes[case].kind, 'experiment')
                self.assertEqual(nodes[case].config['lr'], .1)

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

    def checkpoint_records(self, report, path):
        return next(item for item in report.artifacts if item.path == path).metadata.get('metric_records', [])

    def test_explicit_metric_rows_bind_arbitrary_checkpoint_names(self):
        self.run_folder('run')
        self.put('run/final_final2.pt', b'a')
        self.put('run/new name.ckpt', b'b')
        self.put('run/metrics.csv', 'checkpoint,epoch,accuracy\nfinal_final2.pt,3,0.9\nnew name.ckpt,4,0.8\n')
        report = self.report()
        first = self.checkpoint_records(report, 'run/final_final2.pt')
        second = self.checkpoint_records(report, 'run/new name.ckpt')
        self.assertEqual(first[0]['values'], {'epoch': 3., 'accuracy': .9})
        self.assertEqual(second[0]['values']['epoch'], 4)
        self.assertEqual(first[0]['source'], 'run/metrics.csv:2')
        self.assertGreater(first[0]['confidence'], .9)
        self.assertFalse(self.checkpoint_records(report, 'run/best.pt'))
        self.assertNotIn('_metric_rows', json.dumps(report.to_dict()))

    def test_best_checkpoint_gets_selected_epoch_not_last_epoch(self):
        self.run_folder('run')
        self.put('run/last.pt', b'last')
        self.put('run/result.json', {'best_epoch': 1, 'best_loss': 2, 'initial_loss': 3})
        report = self.report()
        best = self.checkpoint_records(report, 'run/best.pt')
        last = self.checkpoint_records(report, 'run/last.pt')
        self.assertTrue(all(record['values']['epoch'] == 1 for record in best))
        self.assertEqual(last[0]['values'], {'epoch': 2, 'loss': 1})
        self.assertTrue(all(record['confidence'] < .8 for record in best + last))
        self.assertNotIn('initial_loss', best[0]['values'])
        self.assertEqual(report.experiments[0].metrics['history.jsonl:epoch'], 2)

    def test_sidecar_metrics_do_not_become_last_checkpoint_history(self):
        self.run_folder('run')
        self.put('run/initial.pt', b'initial')
        self.put('run/last.pt', b'last')
        self.put('run/initial.json', {'epoch': 0, 'accuracy': .1})
        report = self.report()
        initial = self.checkpoint_records(report, 'run/initial.pt')
        last = self.checkpoint_records(report, 'run/last.pt')
        self.assertEqual(initial[0]['values']['epoch'], 0)
        self.assertEqual(len(last), 1)
        self.assertEqual(last[0]['values']['epoch'], 2)

    def test_best_epoch_conflicts_and_duplicate_rows_are_not_guessed(self):
        self.run_folder('run')
        self.put('run/result.json', {'best_epoch': 1})
        self.put('run/summary.json', {'best_epoch': 2})
        self.assertFalse(self.checkpoint_records(self.report(), 'run/best.pt'))
        self.put('run/summary.json', {'best_epoch': 1})
        self.put('run/history.jsonl', '{"epoch":1,"loss":2}\n{"epoch":1,"loss":1}\n')
        records = self.checkpoint_records(self.report(), 'run/best.pt')
        self.assertTrue(records)
        self.assertTrue(all('history' not in record['source'] for record in records))

    def test_multiple_best_checkpoints_are_ambiguous(self):
        self.run_folder('run')
        self.put('run/best.ckpt', b'other best')
        self.put('run/result.json', {'best_epoch': 1})
        report = self.report()
        self.assertFalse(self.checkpoint_records(report, 'run/best.pt'))
        self.assertFalse(self.checkpoint_records(report, 'run/best.ckpt'))

    def test_last_checkpoint_rejects_partial_malformed_and_restarted_history(self):
        self.run_folder('run')
        self.put('run/last.pt', b'last')
        for history in ('{"epoch":1,"loss":2}\nbroken\n', '{"epoch":2,"loss":1}\n{"epoch":1,"loss":2}\n'):
            self.put('run/history.jsonl', history)
            self.assertFalse(self.checkpoint_records(self.report(), 'run/last.pt'))
        self.put('run/history.jsonl', '{"epoch":1,"loss":2}\n' * 100)
        self.assertFalse(self.checkpoint_records(self.report(text_limit=300), 'run/last.pt'))

    def test_resumed_last_uses_continuation_but_epoch_and_best_keep_prior_history(self):
        self.run_folder('run', resume_from='../base/last.pt')
        self.put('run/last.pt', b'last')
        self.put('run/epoch-2.pt', b'epoch two')
        self.put('run/train.log', 'resume_epoch=2')
        self.put('run/continuation.csv', 'epoch,loss\n3,0.8\n4,0.6\n')
        self.put('run/summary.json', {'best_epoch': 2})
        report = self.report()
        records = self.checkpoint_records(report, 'run/last.pt')
        self.assertEqual([record['values']['epoch'] for record in records], [4])
        self.assertIn('resume_epoch=2', records[0]['basis'])
        self.assertIn('run/train.log', records[0]['basis'])
        self.assertEqual(records[0]['confidence'], .55)
        for path in ('run/best.pt', 'run/epoch-2.pt'):
            self.assertTrue(any(record['source'].startswith('run/history.jsonl') and record['values']['epoch'] == 2
                                for record in self.checkpoint_records(report, path)))
        self.put('run/last.metrics.json', {'epoch': 3, 'loss': .7})
        records = self.checkpoint_records(self.report(), 'run/last.pt')
        self.assertEqual([record['values']['epoch'] for record in records], [3])
        self.assertIn('Matching metadata filename', records[0]['basis'])

    def test_resume_history_requires_unambiguous_adjacent_boundary(self):
        cases = {
            'no-path': ({}, 'resume_epoch=2', '3,0.8\n4,0.6\n'),
            'no-boundary': ({'resume_from': '../base/last.pt'}, '', '3,0.8\n4,0.6\n'),
            'conflicting-paths': ({'resume_from': '../base/last.pt', 'resume': '../other/last.pt'}, 'resume_epoch=2', '3,0.8\n4,0.6\n'),
            'conflicting-boundaries': ({'resume_from': '../base/last.pt', 'resume_epoch': 1}, 'resume_epoch=2', '3,0.8\n4,0.6\n'),
            'overlap': ({'resume_from': '../base/last.pt'}, 'resume_epoch=2', '2,0.8\n4,0.6\n'),
            'gap': ({'resume_from': '../base/last.pt'}, 'resume_epoch=2', '4,0.6\n'),
        }
        for name, (config, log, rows) in cases.items():
            self.run_folder(name, **config)
            self.put(f'{name}/last.pt', b'last')
            self.put(f'{name}/train.log', log)
            self.put(f'{name}/history_old.csv', 'epoch,loss\n' + rows)
        report = self.report()
        for name in cases:
            with self.subTest(case=name):
                records = self.checkpoint_records(report, f'{name}/last.pt')
                self.assertEqual({record['values']['epoch'] for record in records}, {2, 4})
                self.assertTrue(all(record['confidence'] == .4 for record in records))

    def test_metric_record_limit_does_not_imply_final_epoch(self):
        from rundig.metadata import extract, checkpoint_metrics
        from rundig.models import Artifact
        self.put('history.jsonl', '{"epoch":1,"loss":2}\n{"epoch":2,"loss":1}\n{"epoch":3,"loss":0}\n')
        history = Artifact('history', 'history.jsonl', 'metrics', 0, 0)
        checkpoint = Artifact('cp', 'last.pt', 'checkpoint', 0, 0)
        extract(self.root, history, record_limit=2)
        checkpoint_metrics([history, checkpoint])
        self.assertFalse(checkpoint.metadata.get('metric_records'))
        self.assertTrue(history.warnings)

    def test_epoch_filename_needs_a_unique_matching_record(self):
        self.run_folder('run')
        self.put('run/epoch=1.ckpt', b'one')
        self.put('run/final_final2.pt', b'two')
        report = self.report()
        self.assertEqual(self.checkpoint_records(report, 'run/epoch=1.ckpt')[0]['values']['epoch'], 1)
        self.assertFalse(self.checkpoint_records(report, 'run/final_final2.pt'))

    def test_json_metric_records_keep_checkpoint_scopes_separate(self):
        self.run_folder('run')
        self.put('run/other.pt', b'other')
        self.put('run/metrics.json', {'evaluations': [
            {'checkpoint': 'best.pt', 'metrics': {'accuracy': .9}},
            {'checkpoint': 'other.pt', 'metrics': {'accuracy': .2}}]})
        report = self.report()
        self.assertEqual(self.checkpoint_records(report, 'run/best.pt')[0]['values'], {'accuracy': .9})
        self.assertEqual(self.checkpoint_records(report, 'run/other.pt')[0]['values'], {'accuracy': .2})

    def test_input_checkpoint_and_unresolved_output_are_not_metric_targets(self):
        self.run_folder('run')
        self.put('run/other.pt', b'other')
        self.put('run/metrics.json', {'source_checkpoint': 'other.pt', 'accuracy': .9})
        self.put('run/best.json', {'checkpoint': 'missing.pt', 'accuracy': .8})
        report = self.report()
        self.assertFalse(self.checkpoint_records(report, 'run/other.pt'))
        self.assertFalse(self.checkpoint_records(report, 'run/best.pt'))

    def test_nested_evaluation_cannot_leak_into_sidecar_parent(self):
        self.run_folder('run')
        self.put('run/other.pt', b'other')
        self.put('run/best.json', {'evaluation': {'fold1': {'checkpoint': 'other.pt', 'accuracy': .3}}})
        report = self.report()
        self.assertFalse(self.checkpoint_records(report, 'run/best.pt'))
        self.assertEqual(self.checkpoint_records(report, 'run/other.pt')[0]['values'], {'accuracy': .3})

    def test_loose_checkpoint_can_have_metrics_without_an_invented_run(self):
        self.put('new folder/final_final2.pt', b'unknown model')
        self.put('new folder/final_final2.json', {'accuracy': .8})
        report = self.report()
        self.assertEqual(report.experiments[0].kind, 'checkpoint_candidate')
        self.assertEqual(self.checkpoint_records(report, 'new folder/final_final2.pt')[0]['values'], {'accuracy': .8})

    def test_per_sample_predictions_are_not_checkpoint_summary_metrics(self):
        self.run_folder('run')
        self.put('run/best.csv', 'sample,score\n1,0.9\n2,0.1\n')
        self.assertFalse(self.checkpoint_records(self.report(), 'run/best.pt'))

    def test_compare_shows_seed_changes_and_checkpoint_metrics_without_writes(self):
        self.run_folder('a', seed=42)
        self.run_folder('b', seed=123)
        self.put('a/metrics.json', {'checkpoint': 'best.pt', 'accuracy': .9})
        self.put('b/metrics.json', {'checkpoint': 'best.pt', 'accuracy': .8})
        before = sorted(self.root.rglob('*'))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['compare', 'a/best.pt', 'b/best.pt', '--root', str(self.root)]), 0)
        for text in ('train.seed', '42', '123', 'accuracy', '0.9', '0.8', 'a/metrics.json', 'b/metrics.json', 'similar to'):
            self.assertIn(text, output.getvalue())
        self.assertEqual(before, sorted(self.root.rglob('*')))
        self.assertNotIn('train.lr', output.getvalue())

    def test_compare_does_not_substitute_run_metrics_for_unknown_checkpoint(self):
        self.run_folder('a')
        self.run_folder('b')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['compare', str(self.root / 'a/best.pt'), str(self.root / 'b/best.pt')]), 0)
        self.assertIn('Unknown; run-level metrics are not substituted.', output.getvalue())
        self.assertNotIn('loss', output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['compare', 'a/best.pt', 'missing.pt', '--root', str(self.root)]), 2)

    def test_compare_preserves_conflicting_metric_records(self):
        self.run_folder('a')
        self.run_folder('b')
        self.put('a/metrics.json', {'checkpoint': 'best.pt', 'accuracy': .9})
        self.put('a/evaluation.json', {'checkpoint': 'best.pt', 'accuracy': .7})
        self.put('b/metrics.json', {'checkpoint': 'best.pt', 'accuracy': .8})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['compare', 'a/best.pt', 'b/best.pt', '--root', str(self.root), '--full']), 0)
        self.assertIn('Conflicting records; see sources.', output.getvalue())

    def test_inspect_distinguishes_checkpoint_metrics_from_run_summary(self):
        self.run_folder('run')
        self.put('run/result.json', {'best_epoch': 1})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['inspect', 'run/best.pt', '--root', str(self.root)]), 0)
        checkpoint_text = output.getvalue().split('Checkpoint metrics:')[1].split('Run configuration:')[0]
        self.assertIn('epoch: 1', checkpoint_text)
        self.assertNotIn('epoch: 2', checkpoint_text)
        self.assertIn('not necessarily this checkpoint', output.getvalue())

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

    def test_parent_collection_counts_children_without_duplicating_checkpoints(self):
        self.put('batch/config.json', {'lr': .01, 'epochs': 10, 'batch_size': 16})
        self.put('batch/summary.json', {'epoch': 2, 'loss': .2})
        self.run_folder('batch/a')
        self.run_folder('batch/b')
        report = self.report()
        nodes = {node.path: node for node in report.experiments}
        parent = nodes['batch']
        self.assertEqual(parent.kind, 'collection')
        self.assertEqual(parent.checkpoint_count, 0)
        self.assertEqual(parent.subtree_checkpoint_count, 2)
        self.assertEqual(set(parent.children), {nodes['batch/a'].id, nodes['batch/b'].id})
        self.assertEqual(nodes['batch/a'].parent_id, parent.id)
        self.assertEqual(sum(node.checkpoint_count for node in report.experiments), report.statistics['checkpoints'])
        self.assertFalse(any(parent.id in (edge.source, edge.target) and edge.kind == 'similar_to' for edge in report.edges))
        self.assertEqual(report.statistics['collections'], 1)

    def test_nested_collections_have_nearest_parent_and_recursive_totals(self):
        for folder in ('batch', 'batch/nested'):
            self.put(folder + '/config.json', {'lr': .01, 'epochs': 10})
            self.put(folder + '/summary.json', {'loss': .2})
        self.run_folder('batch/nested/run')
        self.run_folder('batch/direct')
        self.run_folder('batch-other')
        nodes = {node.path: node for node in self.report().experiments}
        self.assertEqual(nodes['batch'].subtree_checkpoint_count, 2)
        self.assertEqual(nodes['batch/nested'].subtree_checkpoint_count, 1)
        self.assertEqual(nodes['batch/nested/run'].parent_id, nodes['batch/nested'].id)
        self.assertEqual(nodes['batch/nested'].parent_id, nodes['batch'].id)
        self.assertIsNone(nodes['batch-other'].parent_id)

    def test_checkpoint_storage_folder_is_not_an_experiment_collection(self):
        self.put('run/config.json', {'lr': .1, 'batch_size': 8})
        self.put('run/checkpoints/best.pt', b'best')
        self.put('run/checkpoints/last.pt', b'last')
        report = self.report()
        self.assertEqual(len(report.experiments), 1)
        self.assertEqual(report.experiments[0].kind, 'experiment')
        self.assertEqual(report.experiments[0].checkpoint_count, 2)

    def component_batch(self, name, roles=True, seed=42):
        self.put(name + '/config.json', {'lr': .01, 'batch_size': 16, 'epochs': 10, 'model': 'tiny', 'seed': seed})
        self.put(name + '/summary.json', {'loss': .2, 'phase': 'parent'})
        for child, role in [('odd name', 'audio'), ('another folder', 'vision')]:
            self.put(name + '/' + child + '/best.pt', (name + child).encode())
            self.put(name + '/' + child + '/history.jsonl', '{"epoch":1,"loss":0.3}\n')
            if roles:
                self.put(name + '/' + child + '/state.json', {'phase': role})

    def test_recorded_roles_distinguish_components_with_shared_parent_config(self):
        self.component_batch('batch')
        report = self.report()
        nodes = {node.path: node for node in report.experiments}
        pair = {nodes['batch/odd name'].id, nodes['batch/another folder'].id}
        relation = next(edge for edge in report.edges if {edge.source, edge.target} == pair)
        self.assertEqual(relation.kind, 'similar_to')
        self.assertIn('Different recorded roles', relation.description)
        self.assertLessEqual(relation.confidence, .55)
        self.assertIn('role.phase', relation.changes)
        self.assertEqual(nodes['batch/odd name'].roles['phase'], 'audio')
        self.assertTrue(any(fact.get('artifact', '').endswith('state.json') for fact in relation.evidence))

    def test_shared_configuration_alone_has_low_confidence_and_unknown_roles(self):
        self.component_batch('batch', roles=False)
        report = self.report()
        relations = [edge for edge in report.edges if edge.kind == 'similar_to']
        self.assertEqual(len(relations), 1)
        self.assertIn('run similarity unverified', relations[0].description)
        self.assertLessEqual(relations[0].confidence, .5)
        children = [node for node in report.experiments if node.parent_id]
        self.assertTrue(all(not node.roles for node in children))

    def test_cross_batch_components_are_compared_by_recorded_role(self):
        self.component_batch('batch-a', seed=42)
        self.component_batch('batch-b', seed=3407)
        report = self.report()
        nodes = {node.path: node for node in report.experiments}
        same_role = {nodes['batch-a/odd name'].id, nodes['batch-b/odd name'].id}
        different_roles = {nodes['batch-a/odd name'].id, nodes['batch-b/another folder'].id}
        relation = next(edge for edge in report.edges if {edge.source, edge.target} == same_role)
        self.assertIn('except seed', relation.description)
        self.assertEqual(set(relation.changes['seed'].values()), {42, 3407})
        self.assertLessEqual(relation.confidence, .65)
        self.assertFalse(any({edge.source, edge.target} == different_roles for edge in report.edges))

    def test_similarity_explains_training_and_model_configuration_changes(self):
        self.run_folder('base', seed=42, hidden_size=32)
        self.run_folder('lr-change', seed=42, hidden_size=32, lr=.002)
        self.run_folder('model-change', seed=42, hidden_size=64)
        for name in ('base', 'lr-change', 'model-change'):
            self.put(name + '/source/train.py', '# same training snapshot\n')
        report = self.report()
        nodes = {node.path: node for node in report.experiments}
        for name, label, field in [('lr-change', 'Training settings differ', 'train.lr'), ('model-change', 'Model configuration differs', 'train.hidden_size')]:
            pair = {nodes['base'].id, nodes[name].id}
            relation = next(edge for edge in report.edges if {edge.source, edge.target} == pair)
            self.assertIn(label, relation.description)
            self.assertIn(field, relation.changes)
            self.assertFalse(relation.directed)

    def test_missing_setting_is_not_a_change_or_tentative_ancestry(self):
        self.run_folder('one', hidden_size=32)
        self.run_folder('two')
        for name, date in [('one', '2026-01-01'), ('two', '2026-01-02')]:
            self.put(f'{name}/source/train.py', '# identical code\n')
            self.put(f'{name}/state.json', {'started_at_utc': date + 'T00:00:00Z', 'finished_at_utc': date + 'T01:00:00Z'})
        relation = self.report().edges[0]
        self.assertEqual(relation.kind, 'similar_to')
        self.assertEqual(relation.changes, {})
        self.assertAlmostEqual(relation.confidence, .85)  # Union-based score is unchanged.
        self.assertIn('full configuration uncertain', relation.description)
        unavailable = next(fact['fields'] for fact in relation.evidence if fact['signal'] == 'unavailable comparison fields')
        self.assertEqual(unavailable, ['train.hidden_size'])

    def test_conflicting_settings_on_either_side_are_unknown_not_differences(self):
        for name in ('one', 'two'):
            self.run_folder(name, hidden_size=32, seed=42)
            self.put(f'{name}/source/train.py', '# identical code\n')
            self.put(f'{name}/other_config.json', {'train': {'hidden_size': 64, 'seed': 17}})
        report = self.report()
        relation = report.edges[0]
        self.assertEqual(relation.changes, {})
        self.assertIn('full configuration uncertain', relation.description)
        unavailable = next(fact['fields'] for fact in relation.evidence if fact['signal'] == 'unavailable comparison fields')
        self.assertEqual(set(unavailable), {'train.hidden_size', 'seed'})
        self.assertTrue(all(len(node.config_sources['train.hidden_size']) == 2 for node in report.experiments))

    def test_partial_config_does_not_claim_seed_is_only_difference(self):
        self.run_folder('one', seed=1, hidden_size=32)
        self.run_folder('two', seed=2)
        for name in ('one', 'two'):
            self.put(f'{name}/source/train.py', '# identical code\n')
        relation = self.report().edges[0]
        self.assertEqual(set(relation.changes), {'seed'})
        self.assertIn('configuration partly comparable', relation.description)
        self.assertNotIn('except seed', relation.description)

    def test_confirmed_difference_survives_missing_other_settings(self):
        options = {'dropout': .1, 'layers': 2, 'optimizer': 'adam'}
        self.run_folder('one', hidden_size=32, **options)
        self.run_folder('two', lr=.002, **options)
        for name in ('one', 'two'):
            self.put(f'{name}/source/train.py', '# identical code\n')
        relation = self.report().edges[0]
        self.assertEqual(set(relation.changes), {'train.lr'})
        self.assertIn('Training settings differ', relation.description)
        self.assertTrue(any('Missing or conflicting' in note for note in relation.caveats))

    def test_conflicting_roles_do_not_produce_seed_variant_claim(self):
        self.run_folder('one', seed=1)
        self.run_folder('two', seed=2)
        self.put('one/state.json', {'phase': 'audio'})
        self.put('one/summary.json', {'phase': 'vision'})
        self.put('two/state.json', {'phase': 'audio'})
        report = self.report()
        relation = next(edge for edge in report.edges if edge.kind == 'similar_to')
        self.assertIn('conflicting role declarations', relation.description)
        self.assertLessEqual(relation.confidence, .5)

    def test_seed_change_with_changed_code_is_not_described_as_seed_only(self):
        self.run_folder('one', seed=1)
        self.run_folder('two', seed=2)
        self.put('one/source/train.py', 'MODEL = "first"')
        self.put('two/source/train.py', 'MODEL = "different"')
        relation = next(edge for edge in self.report().edges if edge.kind == 'similar_to')
        self.assertIn('code snapshots also differ', relation.description)
        self.assertNotIn('except seed', relation.description)

    def test_compare_prints_specific_relationship_description(self):
        self.run_folder('one', seed=1)
        self.run_folder('two', seed=2)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['compare', 'one/best.pt', 'two/best.pt', '--root', str(self.root)]), 0)
        self.assertIn('Recorded settings match except seed', output.getvalue())

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

    def test_legacy_report_directory_can_be_updated_after_rename(self):
        self.run_folder('run')
        output = self.root / 'report'
        self.put('report/.run-shelf-output', 'run-shelf report schema 1.0\n')
        self.put('report/index.html', 'old report')
        report = self.report(output=output)
        page = write_report(report, output)
        self.assertTrue((output / '.run-dig-output').is_file())
        self.assertIn('<title>run-dig</title>', page.read_text(encoding='utf-8'))
        self.assertEqual(json.loads((output / 'report.json').read_text(encoding='utf-8')), report.to_dict())
        # Updating an already-renamed report remains valid too.
        write_report(report, output)

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
        with patch('rundig.scan.os.walk', failing_walk):
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

    def test_inspect_prints_each_conflicting_config_value_with_its_source(self):
        self.run_folder('run', seed=42)
        self.put('run/args.json', {'train': {'seed': 2024, 'lr': .001}})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['inspect', str(self.root / 'run/best.pt')]), 0)
        text = output.getvalue()
        self.assertIn('train.seed: Conflicting values', text)
        self.assertIn('2024 | args.json', text)
        self.assertIn('42 | config.json', text)

    def test_conflicting_export_sources_preserve_both_accounts(self):
        self.run_folder('base')
        self.run_folder('old')
        self.run_folder('export', teacher_checkpoint='../old/best.pt')
        self.put('export/source.txt', r'base\best.pt')
        self.put('export/eval.json', {'checkpoint': 'old/best.pt', 'accuracy': .8})
        report = self.report(hash_limit=0)
        run = next(e for e in report.experiments if e.path == 'export')
        incoming = [e for e in report.edges if e.target == run.id and e.directed]
        self.assertEqual(len(incoming), 2)
        self.assertTrue(all(e.kind == 'references' and e.confidence <= .5 for e in incoming))
        self.assertTrue(all(e.description == 'Conflicting source records' for e in incoming))
        self.assertTrue(any('Origin is uncertain' in w for w in run.warnings))
        self.assertEqual({r['artifact'] for r in run.references if r.get('conflict')},
                         {'export/source.txt', 'export/eval.json'})

    def test_matching_export_source_and_self_evaluation_are_not_conflicts(self):
        self.run_folder('base')
        self.run_folder('export')
        self.put('export/source.txt', 'base/best.pt')
        self.put('export/eval.json', {'checkpoint': 'export/best.pt', 'accuracy': .8})
        report = self.report(hash_limit=0)
        self.assertFalse(any('Conflicting source' in w for e in report.experiments for w in e.warnings))
        self.assertEqual([e.confidence for e in report.edges if e.directed], [.9])

    def test_conflicting_resume_configs_lower_confidence(self):
        for name in ['base', 'other']:
            self.run_folder(name)
        self.run_folder('resumed', resume='../base/best.pt')
        self.put('resumed/args.json', {'train': {'resume': '../other/best.pt', 'lr': .001}})
        edges = [e for e in self.report(hash_limit=0).edges if e.directed]
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(e.confidence <= .5 for e in edges))

    def test_wandb_copy_attaches_to_run_without_duplicate_node(self):
        self.run_folder('run', seed=17)
        config = json.loads((self.root / 'run/config.json').read_text())
        self.put('wandb/offline-run-copy/files/config.json', config)
        self.put('wandb/offline-run-copy/files/wandb-summary.json', {'checkpoint': 'run/best.pt', 'accuracy': .9})
        report = self.report(hash_limit=0)
        self.assertEqual(len(report.experiments), 1)
        run = report.experiments[0]
        self.assertEqual(run.path, 'run')
        self.assertEqual(run.checkpoint_count, 1)
        self.assertTrue(any('Tracking records attached' in e for e in run.evidence))
        copied = [a for a in report.artifacts if a.path.startswith('wandb/')]
        self.assertTrue(all(a.id in run.artifacts and a.metadata['tracking_copy_of'] == 'run' for a in copied))
        checkpoint = next(a for a in report.artifacts if a.kind == 'checkpoint')
        self.assertEqual(checkpoint.metadata['metric_records'][0]['source'], copied[1].path)

    def test_wandb_recovers_missing_config_from_compatible_log_facts(self):
        self.put('run/checkpoints/best.pt', b'weights')
        self.put('run/train.log', 'seed=42 hidden=48 lr=0.02\nepoch=1 loss=0.3\n')
        self.put('wandb/offline-run-copy/files/config.json', {'seed': 42, 'hidden': 48, 'lr': .02})
        self.put('wandb/offline-run-copy/files/wandb-summary.json', {'checkpoint': 'run/checkpoints/best.pt', 'accuracy': .9})
        report = self.report(hash_limit=0)
        self.assertEqual(len(report.experiments), 1)
        self.assertEqual(report.experiments[0].config['hidden'], 48)
        self.assertEqual(report.experiments[0].config_sources['hidden'][0]['artifact'], 'wandb/offline-run-copy/files/config.json')

    def test_wandb_conflicting_or_weak_records_stay_separate(self):
        self.run_folder('run', seed=17)
        original = json.loads((self.root / 'run/config.json').read_text())
        for name, config in [('conflict', {**original['train'], 'seed': 42}), ('weak', {'lr': .001, 'seed': 17})]:
            self.put(f'wandb/offline-run-{name}/files/config.json', {'train': config})
            self.put(f'wandb/offline-run-{name}/files/wandb-summary.json', {'checkpoint': 'run/best.pt', 'accuracy': .9})
        report = self.report(hash_limit=0)
        self.assertEqual(len(report.experiments), 3)
        self.assertTrue(all(e.name != 'files' for e in report.experiments))

    def test_code_snapshot_and_one_setting_change_link_arbitrary_names(self):
        self.run_folder('final_final2', balanced=True)
        self.run_folder('new folder', balanced=False)
        for name in ['final_final2', 'new folder']:
            self.put(name + '/code/train.py', 'def train():\n    return 1\n')
        report = self.report(hash_limit=0)
        self.assertTrue(all(e.script_hashes for e in report.experiments))
        relation = next(e for e in report.edges if e.kind == 'similar_to')
        self.assertEqual(set(relation.changes), {'train.balanced'})
        self.assertIn('Training settings differ', relation.description)
        self.assertFalse(any(e.directed for e in report.edges))

    def test_disagreeing_histories_are_flagged_at_checkpoint(self):
        self.put('run/checkpoints/epoch_10.pt', b'weights')
        self.put('run/metrics.csv', 'epoch,loss,accuracy\n10,0.4,0.8\n')
        self.put('run/metrics_old.csv', 'epoch,loss,accuracy\n10,0.7,0.8\n')
        report = self.report(hash_limit=0)
        checkpoint = next(a for a in report.artifacts if a.kind == 'checkpoint')
        self.assertEqual(checkpoint.metadata['metric_conflicts'], ['loss'])
        records = checkpoint.metadata['metric_records']
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r['confidence'] <= .4 for r in records))
        self.assertTrue(any('Metric records disagree' in w for w in checkpoint.warnings))
        self.assertTrue(any('Metric records disagree' in w for w in report.experiments[0].warnings))

    def test_missing_code_is_unknown_rather_than_different_code(self):
        self.run_folder('base', hidden=24, dropout=.1, optimizer='adam', dataset='surface')
        self.run_folder('variant', hidden=48, dropout=.1, optimizer='adam', dataset='surface', lr=.002)
        self.put('base/code/train.py', 'def train():\n    return 1\n')
        relations = self.report(hash_limit=0).edges
        relation = next(e for e in relations if e.kind == 'similar_to')
        self.assertIn('Model configuration differs', relation.description)
        self.assertLessEqual(relation.confidence, .7)
        self.assertTrue(any('Code snapshots are unavailable' in c for c in relation.caveats))
        self.assertFalse(any(e.directed for e in relations))

    def test_metric_rounding_differences_are_not_conflicts(self):
        self.put('run/checkpoints/epoch_10.pt', b'weights')
        self.put('run/metrics.csv', 'epoch,loss\n10,0.333333333333\n')
        self.put('run/metrics_old.csv', 'epoch,loss\n10,0.33333333\n')
        checkpoint = next(a for a in self.report().artifacts if a.kind == 'checkpoint')
        self.assertNotIn('metric_conflicts', checkpoint.metadata)

    def test_mostly_matching_config_with_changed_code_is_still_similar(self):
        common = {'dataset': 'surface', 'hidden': 24, 'dropout': .1, 'optimizer': 'adam', 'save_every': 10, 'selection': 'accuracy'}
        self.run_folder('base', **common, split='random', normalize_all=True)
        self.run_folder('variant', **common, split='group', normalize_all=False)
        self.put('base/code/train.py', 'def train():\n    return 1\n')
        self.put('variant/code/train.py', 'def train():\n    return 2\n')
        relations = self.report(hash_limit=0).edges
        relation = next(e for e in relations if e.kind == 'similar_to')
        self.assertIn('Dataset configuration differs', relation.description)
        self.assertLessEqual(relation.confidence, .7)
        self.assertTrue(any('Code snapshots differ' in c for c in relation.caveats))
        self.assertFalse(any(e.directed for e in relations))

    def test_config_only_similarity_requires_more_than_generic_defaults(self):
        self.run_folder('a', dataset='images', optimizer='adam')
        self.run_folder('b', dataset='speech', optimizer='sgd')
        self.assertFalse(any(e.kind == 'similar_to' for e in self.report(hash_limit=0).edges))

    def test_ids_stable_across_rescan(self):
        self.run_folder("run")
        first, second = self.report(), self.report()
        self.assertEqual(first.experiments, second.experiments)
        self.assertEqual(first.edges, second.edges)

    def test_empty_tree(self):
        self.assertEqual(self.report().statistics["experiments"], 0)

    def test_toml_config_and_invalid_toml(self):
        self.put('run/config.toml', '[train]\nmodel = "tiny"\nlr = 0.001\nbatch_size = 16\n')
        self.put('run/best.pt', b'weights')
        report = self.report()
        self.assertEqual(report.experiments[0].config['train.lr'], .001)
        self.assertEqual(report.experiments[0].config['train.model'], 'tiny')
        self.put('run/config.toml', '[invalid')
        report = self.report()
        self.assertEqual(report.statistics['checkpoints'], 1)
        self.assertTrue(any('parse failed' in note['message'] for note in report.warnings))

    def test_progress_does_not_change_results_or_print_from_library(self):
        self.run_folder('run')
        events = []
        with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
            report = self.report(git=False, progress=lambda *event: events.append(event))
        plain = self.report(git=False).to_dict()
        data = report.to_dict()
        for item in (data, plain):
            item.pop('generated_at')
        self.assertEqual(data, plain)
        self.assertEqual(stdout.getvalue() + stderr.getvalue(), '')
        self.assertIn(('Scanning files', report.statistics['visited_files'], report.statistics['visited_files']), events)
        self.assertIn(('Extracting metadata', len(report.artifacts), len(report.artifacts)), events)
        self.assertIn(('Identifying experiments',), events)
        self.assertIn(('Inferring relationships',), events)

    def test_cli_end_to_end(self):
        self.run_folder("run")
        output = self.root / "report"
        result = subprocess.run([sys.executable, "-m", "rundig", "scan", str(self.root), "-o", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "index.html").is_file())
        self.assertIn("Found 1 checkpoints", result.stdout)
        for stage in ['Scanning files', 'Extracting metadata', 'Identifying experiments', 'Inferring relationships', 'Writing report']:
            self.assertIn(stage, result.stderr)
            self.assertNotIn(stage, result.stdout)

    def test_renamed_module_runs_outside_checkout_with_default_output(self):
        self.run_folder('run')
        result = subprocess.run([sys.executable, '-m', 'rundig', 'scan', '.'], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / 'run-dig-report'
        self.assertTrue((output / '.run-dig-output').is_file())
        self.assertIn('<title>run-dig</title>', (output / 'index.html').read_text(encoding='utf-8'))
        self.assertFalse((self.root / 'run-shelf-report').exists())


if __name__ == "__main__":
    unittest.main()
