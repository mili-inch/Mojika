import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from tests.fixtures import new_state, small_config, step
from train import (
    clip_and_check_gradients,
    epoch_loader,
    full_batches_per_epoch,
    restore_state,
)
from mojika.conditional.metrics import image_loss_metrics


class IndexedDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, sample):
        return torch.tensor([sample.index, sample.mode, sample.visit])


class TrainingBatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_actual_dataset_size_never_produces_single_image_update(self):
        dataset = IndexedDataset(13697)
        training = replace(small_config("adamw").training, batch_size=128)
        loader = epoch_loader(dataset, training, epoch=308, batch_in_epoch=0)
        batches = list(loader)
        self.assertEqual(len(batches), 107)
        self.assertEqual({len(batch) for batch in batches}, {128})
        self.assertEqual(len(torch.unique(torch.cat(batches)[:, 0])), 13696)
        for batch in batches:
            self.assertEqual(torch.bincount(batch[:, 1]).tolist(), [16] * 8)

    def test_image_metrics_expose_a_single_large_error_and_its_condition(self):
        pred = torch.zeros(128, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(pred)
        target[127] = 20
        modes = torch.arange(8).repeat(16)
        metrics = image_loss_metrics(pred, target, modes)
        self.assertEqual(metrics["image_loss_max"], 400)
        self.assertEqual(metrics["image_loss_p99"], 0)
        self.assertEqual(metrics["condition_mean_losses"]["111"], 25)
        self.assertEqual(metrics["condition_mean_losses"]["000"], 0)
        self.assertIsNone(pred.grad)

    def test_resume_keeps_the_remaining_full_batches_and_shuffle(self):
        dataset = IndexedDataset(13697)
        training = replace(small_config("adamw").training, batch_size=128)
        all_batches = list(epoch_loader(dataset, training, 308, 0))
        remaining = list(epoch_loader(dataset, training, 308, 84))
        self.assertEqual(len(remaining), 23)
        for expected, actual in zip(all_batches[84:], remaining, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        parallel = list(
            epoch_loader(dataset, replace(training, num_workers=2), 308, 84)
        )
        for expected, actual in zip(remaining, parallel, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dataset_smaller_than_batch_and_finished_cursor_fail(self):
        with self.assertRaisesRegex(ValueError, "full training batch"):
            full_batches_per_epoch(127, 128)
        dataset = IndexedDataset(7)
        with self.assertRaisesRegex(ValueError, "cursor"):
            epoch_loader(dataset, small_config("adamw").training, 0, 3)

    def test_epoch_mean_uses_only_processed_images(self):
        state = new_state("adamw")
        self.assertIsNone(state.record_batch(torch.tensor(1.0), 2))
        self.assertIsNone(state.record_batch(torch.tensor(2.0), 2))
        self.assertEqual(state.record_batch(torch.tensor(3.0), 2), 2.0)
        self.assertEqual(
            (state.global_step, state.epoch, state.batch_in_epoch), (3, 1, 0)
        )
        self.assertEqual(state.epoch_loss_sum, 0.0)
        with self.assertRaisesRegex(ValueError, "full batch"):
            state.record_batch(torch.tensor(9.0), 1)
        self.assertEqual(state.global_step, 3)

    def test_restored_cursor_is_independent_of_global_optimizer_step(self):
        state = new_state("adamw")
        step(state)
        step(state)
        checkpoint = copy.deepcopy(
            state.checkpoint(torch.get_rng_state(), torch.empty(0, dtype=torch.uint8))
        )
        checkpoint["progress"]["global_step"] = 10000
        checkpoint["progress"]["epoch"] = 308
        restored = restore_state(checkpoint, torch.device("cpu"))
        self.assertEqual(restored.batch_in_epoch, 2)
        expected_mean = (state.epoch_loss_sum + 8.0) / 6
        self.assertEqual(restored.record_batch(torch.tensor(4.0), 2), expected_mean)
        self.assertEqual(restored.global_step, 10001)
        self.assertEqual((restored.epoch, restored.batch_in_epoch), (309, 0))

    def test_checkpoint_cursor_and_epoch_loss_are_validated(self):
        state = new_state("adamw")
        checkpoint = state.checkpoint(
            torch.get_rng_state(), torch.empty(0, dtype=torch.uint8)
        )
        for key, value in (
            ("epoch", -1),
            ("batch_in_epoch", 3),
            ("epoch_loss_sum", 1.0),
        ):
            invalid = copy.deepcopy(checkpoint)
            invalid["progress"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                restore_state(invalid, torch.device("cpu"))

    def test_loss_accumulator_detaches_and_preserves_small_values(self):
        state = new_state("adamw")
        large = torch.tensor(1e8, requires_grad=True)
        state.record_batch(large, 2)
        state.record_batch(torch.tensor(1.0, requires_grad=True), 2)
        self.assertFalse(state.epoch_loss_sum.requires_grad)
        self.assertIsNone(state.epoch_loss_sum.grad_fn)
        self.assertEqual(state.epoch_loss_sum.dtype, torch.float64)
        self.assertEqual(state.epoch_loss_sum.item(), 200000002.0)
        state.optimizers.apply_schedule(state.schedule, state.global_step)
        checkpoint = state.checkpoint(
            torch.get_rng_state(), torch.empty(0, dtype=torch.uint8)
        )
        restored = restore_state(checkpoint, torch.device("cpu"))
        self.assertEqual(restored.record_batch(torch.tensor(1.0), 2), 100000002.0 / 3)

    def test_batches_and_epoch_end_do_not_read_loss_back_to_cpu(self):
        state = new_state("adamw")
        with patch.object(
            torch.Tensor, "item", side_effect=AssertionError("host read")
        ):
            self.assertIsNone(state.record_batch(torch.tensor(1.0), 2))
            self.assertIsNone(state.record_batch(torch.tensor(2.0), 2))
            mean = state.record_batch(torch.tensor(3.0), 2)
        self.assertEqual(mean, 2.0)
        self.assertEqual(mean.dtype, torch.float64)
        self.assertEqual(state.epoch_loss_sum, 0.0)

    def test_gradient_clipping_keeps_separate_bounds_and_checks_every_step(self):
        model, conditioner = (
            torch.nn.Linear(2, 1, bias=False),
            torch.nn.Linear(2, 1, bias=False),
        )
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        conditioner.weight.grad = torch.tensor([[0.1, 0.2]])
        norms = clip_and_check_gradients(
            model, conditioner, 1.0, torch.tensor(1.0), check_loss=False
        )
        self.assertAlmostEqual(norms[0].item(), 5)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[0.6, 0.8]]))
        torch.testing.assert_close(conditioner.weight.grad, torch.tensor([[0.1, 0.2]]))
        for bad in (float("nan"), float("inf")):
            model.weight.grad = torch.full_like(model.weight, bad)
            with self.assertRaises(FloatingPointError):
                clip_and_check_gradients(
                    model, conditioner, 1.0, torch.tensor(1.0), check_loss=False
                )

    def test_loss_check_can_be_scheduled_separately(self):
        model, conditioner = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        for module in (model, conditioner):
            for parameter in module.parameters():
                parameter.grad = torch.zeros_like(parameter)
        loss = torch.tensor(float("inf"))
        clip_and_check_gradients(model, conditioner, 1.0, loss, check_loss=False)
        with self.assertRaises(FloatingPointError):
            clip_and_check_gradients(model, conditioner, 1.0, loss, check_loss=True)


if __name__ == "__main__":
    unittest.main()
