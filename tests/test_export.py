import tempfile
import unittest
from pathlib import Path

import torch

from mojika.conditional.checkpoint import load_models, save_checkpoint
from mojika.conditional.export import export_weights
from tests.fixtures import new_state, step


class ExportTests(unittest.TestCase):
    def test_export_preserves_predictions_without_training_metadata(self):
        torch.set_num_threads(2)
        state = new_state("adamw")
        for _ in range(3):
            step(state)
        checkpoint = state.checkpoint(torch.get_rng_state(), torch.get_rng_state())
        checkpoint["dataset"]["root"] = "/private/dataset"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "training.pt"
            destination = Path(directory) / "weights.pt"
            save_checkpoint(source, checkpoint)
            export_weights(source, destination)
            exported = torch.load(destination, weights_only=True)
            self.assertEqual(
                set(exported),
                {"config", "model_state_dict", "conditioner_state_dict"},
            )
            model, conditioner = load_models(exported)
            x, t = torch.randn(1, 1, 256, 256), torch.rand(1, 16, 16)
            conditions = {"gpt": torch.randn(1, 1280), "clip": torch.randn(1, 512)}
            for name in ("gpt", "clip"):
                torch.testing.assert_close(
                    getattr(conditioner, f"{name}_embedder").mean,
                    getattr(state.conditioner, f"{name}_embedder").mean,
                    rtol=0,
                    atol=0,
                )
            with torch.inference_mode():
                expected = state.model(x, t, *state.conditioner(x, **conditions))
                actual = model(x, t, *conditioner(x, **conditions))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(
                torch.load(source, weights_only=True)["dataset"]["root"],
                "/private/dataset",
            )
            before = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                export_weights(source, destination)
            self.assertEqual(destination.read_bytes(), before)

    def test_invalid_model_does_not_produce_an_export(self):
        state = new_state("adamw")
        checkpoint = state.checkpoint(torch.get_rng_state(), torch.get_rng_state())
        del checkpoint["model_state_dict"]["proj_out.weight"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "training.pt"
            destination = Path(directory) / "weights.pt"
            save_checkpoint(source, checkpoint)
            with self.assertRaises(RuntimeError):
                export_weights(source, destination)
            self.assertFalse(destination.exists())
