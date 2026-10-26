"""Small real repositories exercise Git integration without a network or ML libraries."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rundig.cli import main
from rundig.git_history import GitReader
from rundig.metadata import extract
from rundig.report import write_report
from rundig.scan import analyze


@unittest.skipUnless(shutil.which("git"), "Git is optional")
class GitHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
                        GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid")
        self.git("init", "-q", "-b", "main")

    def tearDown(self):
        # Git object files can be read-only on Windows.
        for folder, _, files in os.walk(self.root):
            for name in files:
                (Path(folder) / name).chmod(0o600)
        self.temporary.cleanup()

    def git(self, *args, folder=None):
        return subprocess.run(["git", "-C", str(folder or self.root), *args], env=self.env,
                              capture_output=True, check=True).stdout.decode("utf-8").strip()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")
        return path

    def commit(self, message, folder=None):
        self.git("add", ".", folder=folder)
        self.git("-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + os.devnull,
                 "commit", "-qm", message, folder=folder)
        return self.git("rev-parse", "HEAD", folder=folder)

    def run_folder(self, name, revision=None, **extra):
        config = {"model": "tiny", "dataset": "sample", "lr": .001, "batch_size": 16, **extra}
        if revision:
            config["git_commit"] = revision
        self.put(name + "/config.json", config)
        self.put(name + "/best.pt", "placeholder " + name)

    def test_history_is_scoped_to_owned_files_and_not_training_version(self):
        self.run_folder("new folder [1]")
        first = self.commit("initial run")
        self.put("unrelated.txt", "unrelated")
        latest = self.commit("unrelated change")
        before = (self.root / ".git/index").read_bytes()
        report = analyze(self.root)
        data = report.experiments[0].git
        self.assertEqual(data["head"]["commit"], latest)
        self.assertEqual(data["branch"], "main")
        self.assertEqual([item["commit"] for item in data["history"]], [first])
        self.assertEqual(data["history_scope"], "files")
        self.assertEqual(data["recorded_commits"], [])
        self.assertEqual(before, (self.root / ".git/index").read_bytes())
        self.assertEqual(report.edges, [])

    def test_recorded_ancestor_only_annotates_existing_relationship(self):
        self.put("train.py", "width = 16\n")
        first = self.commit("initial model")
        self.put("train.py", "width = 32\n")
        second = self.commit("change module")
        self.run_folder("base", first[:12])
        self.run_folder("retry", second, resume_from="../base/best.pt")
        report = analyze(self.root)
        plain = analyze(self.root, git=False)
        self.assertEqual([(edge.kind, edge.confidence) for edge in report.edges],
                         [(edge.kind, edge.confidence) for edge in plain.edges])
        evidence = [fact for edge in report.edges for fact in edge.evidence if fact['signal'] == 'recorded Git versions']
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["value"], "first recorded commit is an ancestor of the second")
        self.assertEqual(evidence[0]["source_commit"], first)
        self.assertEqual(evidence[0]["target_commit"], second)
        self.assertEqual(report.experiments[0].git["history_scope"], "repository")  # untracked runs
        self.assertEqual(report.experiments[0].git["history"][0]["commit"], second)

    def test_same_commit_and_shared_repository_do_not_create_edges(self):
        self.put("train.py", "pass\n")
        revision = self.commit("model")
        self.run_folder("a", revision, model="one", dataset="a", lr=1, batch_size=1)
        self.run_folder("b", revision, model="two", dataset="b", lr=2, batch_size=2)
        self.assertEqual(analyze(self.root).edges, [])

    def test_git_fields_do_not_count_as_config_agreement(self):
        self.put("train.py", "pass\n")
        revision = self.commit("model")
        for name in ("a", "b"):
            self.put(f"{name}/config.json", {"model": "tiny", "lr": .1, "git_commit": revision})
            self.put(f"{name}/best.pt", name)
        self.assertEqual(analyze(self.root).edges, [])

    def test_conflicting_log_records_are_retained_and_not_used_for_edges(self):
        self.put("train.py", "pass\n")
        first = self.commit("first")
        self.put("train.py", "x = 1\n")
        second = self.commit("second")
        self.run_folder("a")
        self.put("a/train.log", f"git_commit={first}\nGit commit: {second}\n")
        self.run_folder("b", second)
        report = analyze(self.root)
        data = next(item.git for item in report.experiments if item.path == "a")
        self.assertEqual({record["commit"] for record in data["recorded_commits"]}, {first, second})
        self.assertEqual([record['line'] for record in data['recorded_commits']], [1, 2])
        self.assertTrue(any("Conflicting" in note for note in data["notes"]))
        self.assertTrue(report.edges)
        self.assertFalse(any(fact['signal'] == 'recorded Git versions' for edge in report.edges for fact in edge.evidence))

    def test_unverified_revisions_and_shared_parent_config(self):
        self.put("config.json", {"lr": .001, "epochs": 10, "git_commit": "a" * 40})
        self.put("a/best.pt", "a")
        self.put("a/train.log", "epoch=1 loss=0.5")
        self.run_folder("b", "--help")
        report = analyze(self.root)
        a, b = (next(item for item in report.experiments if item.path == path) for path in ("a", "b"))
        self.assertEqual(a.git['recorded_commits'], [])  # no borrowed training commit
        self.assertEqual(b.git['recorded_commits'][0]['resolution'], 'unverified')
        self.assertIsNone(b.git['head'])

    def test_subdirectory_scan_detached_head_and_nested_repository(self):
        self.run_folder("project/run")
        outer = self.commit("outer")
        self.git("checkout", "--detach", outer)
        nested = self.root / "nested"
        nested.mkdir()
        self.git("init", "-q", "-b", "other", folder=nested)
        self.run_folder("nested/run")
        inner = self.commit("inner", folder=nested)
        report = analyze(self.root)
        data = {item.path: item.git for item in report.experiments}
        self.assertEqual(data["nested/run"]["head"]["commit"], inner)
        self.assertEqual(data["project/run"]["head"]["commit"], outer)
        self.assertIsNone(data["project/run"]["branch"])
        sub = analyze(self.root / "project/run").experiments[0].git
        self.assertEqual(sub['repository'], '../..')
        self.assertEqual(sub['history'][0]['commit'], outer)

    def test_worktree_git_file(self):
        self.run_folder("run")
        revision = self.commit("initial")
        self.git("worktree", "add", "--detach", str(self.root / "linked"), revision)
        data = analyze(self.root / "linked").experiments[0].git
        self.assertEqual(data['head']['commit'], revision)
        self.assertEqual(data['repository'], '.')

    def test_shallow_history_does_not_assert_divergence(self):
        self.put("train.py", "pass\n")
        first = self.commit("first")
        self.put("train.py", "x = 1\n")
        second = self.commit("second")
        clone = self.root / "shallow"
        self.git("clone", "--depth", "1", self.root.as_uri(), str(clone))
        self.run_folder("shallow/run", first)
        report = analyze(clone)
        data = report.experiments[0].git
        self.assertTrue(data['shallow'])
        self.assertEqual(data['head']['commit'], second)
        self.assertEqual(data['recorded_commits'][0]['resolution'], 'unverified')
        reader = GitReader(clone, [])
        self.assertIsNone(reader.relation(clone, first, second))

    def test_missing_git_and_timeout_keep_scan_usable(self):
        self.run_folder("run", "a" * 40)
        with patch('rundig.git_history.shutil.which', return_value=None):
            report = analyze(self.root)
        self.assertEqual(report.statistics['checkpoints'], 1)
        self.assertFalse(report.statistics['partial'])
        self.assertTrue(any('not installed' in note['message'] for note in report.warnings))
        with patch('rundig.git_history.subprocess.run', side_effect=subprocess.TimeoutExpired('git', 3)):
            report = analyze(self.root)
        self.assertEqual(report.experiments[0].git['recorded_commits'][0]['resolution'], 'unverified')

    def test_no_git_flag_and_cli_output(self):
        self.run_folder("run")
        revision = self.commit("record run")
        self.put("run/metadata.json", {'git': {'commit': revision}})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(['inspect', 'run/best.pt', '--root', str(self.root)])
        self.assertEqual(result, 0)
        self.assertIn('Git:', output.getvalue())
        self.assertIn('record run', output.getvalue())
        self.assertIn('run/metadata.json#git.commit', output.getvalue())
        with patch('rundig.scan.attach_git', side_effect=AssertionError('must not inspect Git')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['inspect', 'run/best.pt', '--root', str(self.root), '--no-git']), 0)
                self.assertEqual(main(['scan', str(self.root), '-o', str(self.root / 'report'), '--no-git']), 0)

    def test_subjects_are_data_and_reports_contain_git_context(self):
        self.run_folder("run")
        self.commit('</script><img src=x onerror=alert(1)>')
        report = analyze(self.root)
        html = write_report(report, self.root / 'report').read_text(encoding='utf-8')
        self.assertNotIn('</script><img', html)
        data = json.loads((self.root / 'report/report.json').read_text(encoding='utf-8'))
        self.assertEqual(data['experiments'][0]['git']['history'][0]['summary'], '</script><img src=x onerror=alert(1)>')

    def test_nested_component_commit_is_not_the_run_commit(self):
        self.put('train.py', 'pass\n')
        revision = self.commit('model')
        self.run_folder('run')
        self.put('run/metadata.json', {'teacher': {'git': {'commit': revision}},
                                       'train': {'git_commit': revision}})
        self.put('run/train.log', f'teacher.git.commit={revision}\n')
        records = analyze(self.root).experiments[0].git['recorded_commits']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['field'], 'train.git_commit')

    def test_numeric_prefix_in_log_is_not_a_numeric_commit_record(self):
        self.run_folder('run')
        self.put('run/train.log', 'git_commit=123abcde\n')
        records = analyze(self.root).experiments[0].git['recorded_commits']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['value'], '123abcde')
        self.assertEqual(records[0]['resolution'], 'unverified')

    def test_structured_logs_preserve_conflicting_commits_and_lines(self):
        self.put('train.py', 'pass\n')
        first = self.commit('first')
        self.put('train.py', 'x = 1\n')
        second = self.commit('second')
        self.run_folder('other', second)
        for suffix, separator in [('jsonl', None), ('csv', ','), ('tsv', '\t')]:
            self.run_folder(suffix)
            text = ('\n'.join(json.dumps({'git_commit': value, 'epoch': i}) for i, value in enumerate([first, second]))
                    if separator is None else f'git_commit{separator}epoch\n{first}{separator}0\n{second}{separator}1\n')
            self.put(f'{suffix}/history.{suffix}', text)
        report = analyze(self.root)
        for suffix in ['jsonl', 'csv', 'tsv']:
            run = next(item for item in report.experiments if item.path == suffix)
            records = run.git['recorded_commits']
            self.assertEqual([r['commit'] for r in records], [first, second])
            self.assertEqual([r['line'] for r in records], [1, 2] if suffix == 'jsonl' else [2, 3])
            self.assertTrue(any('Conflicting' in note for note in run.git['notes']))
            self.assertFalse(any(f['signal'] == 'recorded Git versions' for edge in report.edges
                                 if run.id in {edge.source, edge.target} for f in edge.evidence))
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['inspect', 'jsonl/best.pt', '--root', str(self.root)]), 0)
        self.assertIn('jsonl/history.jsonl:1#git_commit', output.getvalue())
        self.assertIn('jsonl/history.jsonl:2#git_commit', output.getvalue())

    def test_incomplete_git_log_cannot_supply_relationship_evidence(self):
        self.put('train.py', 'pass\n')
        revision = self.commit('model')
        self.run_folder('run')
        self.run_folder('other', revision)
        line = json.dumps({'git_commit': revision}) + '\n'
        self.put('run/train.jsonl', line + '{invalid json}\n')
        report = analyze(self.root)
        run = next(item for item in report.experiments if item.path == 'run')
        self.assertTrue(any('Incomplete Git records' in note for note in run.git['notes']))
        self.assertFalse(any(f['signal'] == 'recorded Git versions' for edge in report.edges for f in edge.evidence))
        self.put('run/train.jsonl', line * 3)
        artifact = next(item for item in report.artifacts if item.path == 'run/train.jsonl')
        for limits in ({'record_limit': 1}, {'text_limit': len(line) + 5}):
            with self.subTest(limits=limits):
                extract(self.root, artifact, **limits)
                self.assertFalse(artifact.metadata['git_records_complete'])
                self.assertEqual(len(artifact.metadata['git_commits']), 1)

    def test_unavailable_commits_still_show_conflicts_but_prefixes_can_agree(self):
        self.run_folder('run')
        for revisions, conflict in [(['a' * 40, 'b' * 40], True), (['a' * 12, 'a' * 40], False)]:
            with self.subTest(revisions=revisions):
                self.put('run/train.jsonl', '\n'.join(json.dumps({'git_commit': value}) for value in revisions))
                data = analyze(self.root).experiments[0].git
                self.assertEqual([r['value'] for r in data['recorded_commits']], revisions)
                self.assertTrue(all(r['resolution'] == 'unverified' for r in data['recorded_commits']))
                self.assertEqual(any('Conflicting' in note for note in data['notes']), conflict)

    def test_default_cli_hides_checkout_history_but_full_keeps_it(self):
        self.run_folder('run')
        self.commit('history-only-message')
        for command in (['inspect', 'run/best.pt'], ['compare', 'run/best.pt', 'run/best.pt']):
            for full in (False, True):
                with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                    code = main([*command, '--root', str(self.root), *(['--full'] if full else [])])
                self.assertEqual(code, 0)
                self.assertIn('Training commit: unknown', output.getvalue())
                self.assertEqual('history-only-message' in output.getvalue(), full)
                self.assertEqual('Current checkout:' in output.getvalue(), full)


if __name__ == '__main__':
    unittest.main()
