"""Protocol regressions with analytically known rankings; no GPU or CLIP."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_protocol import audit_export, protocol_metrics, run


class ProtocolAuditTests(unittest.TestCase):
    def test_macro_and_pooled_are_distinct(self):
        result = protocol_metrics([[0.2, 0.1], [0.9, 0.8]], [[1, 0], [1, 0]])
        self.assertEqual(result["macro_auc_defined_samples"], 1)
        self.assertEqual(result["pooled_auc"], 0.75)

    def test_exact_half_labels_change_auc_not_mae(self):
        p, y = [[0.9, 0.8, 0.1]], [[0.5, 1, 0]]
        strict, inclusive = protocol_metrics(p, y), protocol_metrics(p, y, True)
        self.assertEqual(strict["pooled_auc"], 0.5)
        self.assertEqual(inclusive["pooled_auc"], 1)
        self.assertEqual(strict["mae"], inclusive["mae"])
        self.assertEqual(inclusive["exact_half_label_points"], 1)

    def test_undefined_auc_explicit_and_invalid_inputs_fail(self):
        result = protocol_metrics([[0.2, 0.3], [0.8, 0.9]], [[0, 0], [1, 1]], True)
        self.assertEqual(result["undefined_auc_samples"], 2)
        self.assertIsNone(result["macro_auc_defined_samples"])
        self.assertFalse(result["official_overall_auc_comparable"])
        self.assertEqual(result["pooled_auc"], 1)
        for p, y in [([], []), ([[np.nan]], [[0]]), ([[2]], [[0]]), ([[0]], [[0, 1]])]:
            with self.assertRaises(ValueError):
                protocol_metrics(p, y)

    def fixture_export(self, directory):
        directory.mkdir(parents=True)
        np.savez(directory / "predictions.npz", probabilities=[[0.8, 0.2]], masks=[[1., 0.]],
                 shape_ids=["shape_a"], queries=["open?"])
        meta = {"split": "val", "dataset": "laso", "samples": 1, "num_points": 2,
                "pooled": {"auc": 1., "mae": 0.2}, "checkpoint_sha256": "fixture",
                "data_hashes": {}, "query_mode": "official", "seed": 42}
        (directory / "metrics.json").write_text(json.dumps(meta))
        return meta

    def test_export_provenance_and_test_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "export"
            meta = self.fixture_export(directory)
            result = audit_export(directory)
            self.assertEqual(result["unique_shapes"], 1)
            self.assertEqual(len(result["predictions_sha256"]), 64)
            meta["split"] = "test"
            (directory / "metrics.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "val"):
                audit_export(directory)
            meta["split"] = "val"
            meta["pooled"]["mae"] = 0.9
            (directory / "metrics.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "MAE"):
                audit_export(directory)

    def test_audit_no_training_no_mutation_and_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "stage2", Path(temp) / "audit"
            root.mkdir()
            (root / "status.json").write_text('{"status":"complete"}')
            (root / "stage2_plan.json").write_text('{}')
            for name in ("legacy_bce", "local_bce", "query_local_bce"):
                directory = root / "structure" / name
                directory.mkdir(parents=True)
                (directory / "history.json").write_text(json.dumps([
                    {"epoch": 1, "auc": 0.7, "macro_auc": 0.8, "mae": 0.2},
                    {"epoch": 2, "auc": 0.8, "macro_auc": 0.7, "mae": 0.1}]))
            self.fixture_export(root / "calibration_diagnostics/bce/validation_export")
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            args = argparse.Namespace(stage2_root=root, output_dir=out, export_structures=False)
            with patch('audit_protocol.subprocess.run') as child, contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
                child.assert_not_called()
            self.assertEqual(report["histories"][0]["highest_strict_macro_auc"]["epoch"], 1)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                run(args)
            # Export requests stay on val, preserve model files, and deduplicate
            # identical best_joint/best_auc checkpoint bytes.
            args.output_dir = Path(temp) / "gpu_audit"
            args.export_structures = True
            args.gpu, args.data_dir = "1", str(Path(temp) / "data")
            for name in ("legacy_bce", "local_bce", "query_local_bce"):
                for criterion in ("joint", "auc"):
                    (root / "structure" / name / f"best_{criterion}.pth").write_bytes(name.encode())
            import audit_protocol
            def exported(command, **kwargs):
                self.assertEqual(command[command.index('--split') + 1], 'val')
                directory = Path(command[command.index('--output-dir') + 1])
                checkpoint = Path(command[command.index('--checkpoint') + 1])
                meta = self.fixture_export(directory)
                meta['checkpoint_sha256'] = audit_protocol.digest(checkpoint)
                (directory / 'metrics.json').write_text(json.dumps(meta))
            with patch('audit_protocol.subprocess.run', side_effect=exported) as child, contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(child.call_count, 3)
            self.assertEqual(len(report['checkpoint_exports']), 6)
            args.output_dir = root / "audit"
            with self.assertRaisesRegex(ValueError, "separate"):
                run(args)


if __name__ == "__main__":
    unittest.main()
