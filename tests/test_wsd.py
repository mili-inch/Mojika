import copy
import itertools
import tempfile
import unittest
from pathlib import Path

import torch

from mojika.conditional.checkpoint import save_checkpoint
from mojika.conditional.config import ExperimentConfig
from mojika.conditional.schedule import WSD
from tests.fixtures import new_state, small_config, step
from train import fork_decay, parse_args, restore_state


class ScheduleTests(unittest.TestCase):
    def test_warmup_has_exact_boundaries_and_stable_has_no_end(self):
        schedule = WSD(4, (0.02, 0.0003), None)
        self.assertEqual(
            [schedule.multiplier(i) for i in range(7)], [0, 0.25, 0.5, 0.75, 1, 1, 1]
        )
        self.assertEqual(schedule.learning_rates(10**12), schedule.base_lrs)
        self.assertIsNone(schedule.end_step)
        self.assertEqual(schedule.phase(3), "warmup")
        self.assertEqual(schedule.phase(4), "stable")

    def test_no_warmup(self):
        schedule = WSD(0, (0.1,), None)
        self.assertEqual(schedule.learning_rates(0), (0.1,))

    def test_decay_curves_preserve_group_ratio_and_reach_endpoint(self):
        parent = WSD(2, (0.02, 0.0003), None)
        for curve in ("linear", "sqrt", "cosine"):
            with self.subTest(curve=curve):
                child = parent.fork_decay(100, 10, curve, 0.01)
                values = [child.multiplier(step) for step in range(100, 111)]
                self.assertEqual(values[0], 1)
                self.assertEqual(values[-1], 0.01)
                self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
                a, b = child.learning_rates(105)
                self.assertAlmostEqual(a / b, 0.02 / 0.0003)
                restored = WSD.from_dict(child.to_dict())
                self.assertEqual(
                    restored.learning_rates(106), child.learning_rates(106)
                )
                self.assertEqual(child.end_step, 110)
                self.assertEqual(child.phase(110), "complete")
        self.assertIsNone(parent.decay)

    def test_decay_rejects_unfinished_warmup_and_rebranching(self):
        parent = WSD(4, (0.001,), None)
        with self.assertRaisesRegex(ValueError, "Stable"):
            parent.fork_decay(3, 10, "linear", 0)
        child = parent.fork_decay(4, 10, "linear", 0)
        with self.assertRaisesRegex(ValueError, "Stable"):
            child.fork_decay(6, 10, "linear", 0)
        for invalid in (3, 15):
            with self.assertRaisesRegex(ValueError, "outside"):
                child.learning_rates(invalid)

    def test_invalid_decay_arguments_fail(self):
        parent = WSD(0, (0.001,), None)
        for steps, curve, minimum in (
            (0, "linear", 0),
            (2, "unknown", 0),
            (2, "sqrt", float("nan")),
            (2, "cosine", -1),
        ):
            with self.assertRaises(ValueError):
                parent.fork_decay(10, steps, curve, minimum)


class OptimizerAndResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def payload(self, state):
        payload = state.checkpoint(
            torch.get_rng_state(), torch.empty(0, dtype=torch.uint8)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, payload)
            return torch.load(path, weights_only=True)

    def test_muon_only_owns_hidden_matrices(self):
        state = new_state("muon_adamw")
        layout = state.optimizers.layout
        self.assertEqual(len(layout["muon"]), 5)
        self.assertTrue(
            all(
                name.startswith("model.blocks.") and name.endswith(".weight")
                for name in layout["muon"]
            )
        )
        self.assertIn("model.blocks.0.attn.rope.raw_freqs", layout["adamw"])
        self.assertIn("model.final_modulation.time.scale.raw_slope", layout["adamw"])
        no_decay = next(
            group
            for group in state.optimizers.groups()
            if group["name"] == "adamw_no_decay"
        )
        self.assertEqual(no_decay["weight_decay"], 0)
        no_decay_ids = {id(parameter) for parameter in no_decay["params"]}
        self.assertIn(
            id(state.model.final_modulation.time["scale"].raw_slope), no_decay_ids
        )
        self.assertIn(id(state.model.blocks[0].attn.rope.raw_freqs), no_decay_ids)
        self.assertIn("model.final_modulation.time.scale.param0", layout["adamw"])
        self.assertIn("model.proj_out.weight", layout["adamw"])
        self.assertIn("model.patch_embed.weight", layout["adamw"])
        self.assertTrue(
            all(
                name in layout["adamw"]
                for name, _ in [
                    (f"conditioner.{n}", p)
                    for n, p in state.conditioner.named_parameters()
                ]
            )
        )
        self.assertFalse(set(layout["muon"]) & set(layout["adamw"]))

    def test_unused_condition_parameters_skip_adamw_update(self):
        state = new_state("muon_adamw")
        state.optimizers.set_learning_rates(state.optimizers.base_lrs)
        for module in (state.model, state.conditioner):
            for parameter in module.parameters():
                parameter.grad = torch.ones_like(parameter)
        state.optimizers.step()
        parameter = state.conditioner.gpt_embedder.linear.weight
        previous = parameter.detach().clone()
        adam = state.optimizers.items["adamw"]
        previous_step = adam.state[parameter]["step"]
        state.optimizers.zero_grad()
        for parameter_hidden in state.optimizers.items["muon"].param_groups[0][
            "params"
        ]:
            parameter_hidden.grad = torch.zeros_like(parameter_hidden)
        state.optimizers.step()
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
        self.assertEqual(adam.state[parameter]["step"], previous_step)

    def test_uninterrupted_and_resumed_next_update_are_identical(self):
        for algorithm, phase in itertools.product(
            ("adamw", "muon_adamw"), ("warmup", "stable", "decay")
        ):
            with self.subTest(algorithm=algorithm, phase=phase):
                state = new_state(algorithm)
                for _ in range(1 if phase == "warmup" else 3):
                    step(state)
                if phase == "decay":
                    fork_decay(state, Path("stable.pt"), 8, "cosine", 0.01)
                    step(state)
                checkpoint = self.payload(state)
                expected_loss = step(state)
                restored = restore_state(checkpoint, torch.device("cpu"))
                torch.set_rng_state(checkpoint["rng_state"])
                actual_loss = step(restored)
                self.assertEqual(actual_loss, expected_loss)
                self.assertEqual(restored.global_step, state.global_step)
                self.assertEqual(restored.epoch, state.epoch)
                self.assertEqual(restored.batch_in_epoch, state.batch_in_epoch)
                self.assertEqual(restored.epoch_loss_sum, state.epoch_loss_sum)
                self.assertEqual(restored.schedule, state.schedule)
                for expected, actual in zip(
                    itertools.chain(
                        state.model.parameters(), state.conditioner.parameters()
                    ),
                    itertools.chain(
                        restored.model.parameters(), restored.conditioner.parameters()
                    ),
                    strict=True,
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_decay_fork_preserves_parent_and_moments(self):
        parent = new_state("muon_adamw")
        for _ in range(3):
            step(parent)
        checkpoint = self.payload(parent)
        child = restore_state(checkpoint, torch.device("cpu"))
        moments = copy.deepcopy(child.optimizers.state_dict())
        fork_decay(child, Path("stable.pt"), 5, "linear", 0)
        self.assertIsNone(parent.schedule.decay)
        self.assertEqual(child.global_step, parent.global_step)
        self.assertEqual(child.schedule.decay.start_step, parent.global_step)
        after = child.optimizers.state_dict()
        for name in moments["states"]:
            for parameter_id, buffers in moments["states"][name]["state"].items():
                for key, value in buffers.items():
                    torch.testing.assert_close(
                        after["states"][name]["state"][parameter_id][key],
                        value,
                        rtol=0,
                        atol=0,
                    )

    def test_corrupt_learning_rate_or_parameter_assignment_is_rejected(self):
        state = new_state("adamw")
        step(state)
        checkpoint = self.payload(state)
        wrong_rate = copy.deepcopy(checkpoint)
        wrong_rate["optimizer_state_dict"]["states"]["adamw"]["param_groups"][0][
            "lr"
        ] = 0.9
        with self.assertRaisesRegex(ValueError, "learning rates"):
            restore_state(wrong_rate, torch.device("cpu"))
        checkpoint["optimizer_state_dict"]["layout"]["adamw"].reverse()
        with self.assertRaisesRegex(ValueError, "assignment"):
            restore_state(checkpoint, torch.device("cpu"))

    def test_momentum_or_weight_decay_disagreeing_with_progress_is_rejected(self):
        state = new_state("muon_adamw")
        step(state)
        for field in ("momentum", "weight_decay"):
            checkpoint = self.payload(state)
            checkpoint["optimizer_state_dict"]["states"]["muon"]["param_groups"][0][
                field
            ] = 0.5
            with self.assertRaisesRegex(ValueError, "scheduler progress"):
                restore_state(checkpoint, torch.device("cpu"))


class ConfigurationTests(unittest.TestCase):
    def test_time_sampling_round_trip_requires_all_parameters(self):
        config = small_config("adamw")
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)
        parameters = config.to_dict()["training"]["time_sampling"]
        self.assertEqual(
            parameters,
            {
                "location": -0.8,
                "scale": 0.8,
                "noise_density": 0.2,
                "image_density": 0.2,
            },
        )
        for field in parameters:
            data = config.to_dict()
            del data["training"]["time_sampling"][field]
            with self.assertRaises(TypeError):
                ExperimentConfig.from_dict(data)
        data = config.to_dict()
        del data["training"]["time_sampling"]
        with self.assertRaises(KeyError):
            ExperimentConfig.from_dict(data)
        data = config.to_dict()
        data["training"]["time_sampling"]["noise_density"] = 2
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict(data)

    def test_loss_check_interval_must_be_explicit_and_positive(self):
        config = small_config("adamw").to_dict()
        for value in (0, -1, True):
            config["training"]["check_loss_every_steps"] = value
            with self.assertRaisesRegex(ValueError, "check_loss_every_steps"):
                ExperimentConfig.from_dict(config)
        del config["training"]["check_loss_every_steps"]
        with self.assertRaises(TypeError):
            ExperimentConfig.from_dict(config)

    def test_optimizer_selection_requires_its_own_settings(self):
        adam = small_config("adamw").to_dict()
        self.assertEqual(ExperimentConfig.from_dict(adam).optimizer.name, "adamw")
        adam["optimizer"]["name"] = "muon_adamw"
        with self.assertRaises(TypeError):
            ExperimentConfig.from_dict(adam)

    def test_missing_or_unknown_configuration_is_rejected(self):
        config = small_config("adamw").to_dict()
        del config["warmup_steps"]
        with self.assertRaisesRegex(ValueError, "sections"):
            ExperimentConfig.from_dict(config)
        config = small_config("adamw").to_dict()
        config["optimizer"]["name"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown optimizer"):
            ExperimentConfig.from_dict(config)

    def test_stable_cli_does_not_require_a_total_training_duration(self):
        args = parse_args(
            [
                "stable",
                "--config",
                "experiment.toml",
                "--dataset-root",
                "data",
                "--output-dir",
                "run",
            ]
        )
        self.assertIsNone(args.max_steps)


if __name__ == "__main__":
    unittest.main()
