import itertools
import unittest
from collections import Counter, defaultdict

import torch

from mojika.conditional.condition_sampling import (
    ConditionSample,
    embedding_variants,
    epoch_samples,
)


class ConditionSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_production_batches_balance_every_mode_and_drop_one_image(self):
        samples = epoch_samples(13697, 128, 42, 308)
        self.assertEqual(len(samples), 13696)
        self.assertEqual(len({s.index for s in samples}), 13696)
        for offset in range(0, len(samples), 128):
            self.assertEqual(
                Counter(s.mode for s in samples[offset : offset + 128]),
                {mode: 16 for mode in range(8)},
            )

    def test_batch_companions_are_remixed_throughout_the_condition_cycle(self):
        for batch_size in (8, 16, 128):
            with self.subTest(batch_size=batch_size):
                size = batch_size * 32 + 1
                companions = {}
                included = None
                for epoch in range(8):
                    samples = epoch_samples(size, batch_size, 42, epoch)
                    indices = {sample.index for sample in samples}
                    self.assertEqual(len(samples), size - 1)
                    self.assertEqual(len(indices), len(samples))
                    if included is None:
                        included = indices
                    self.assertEqual(indices, included)
                    for offset in range(0, len(samples), batch_size):
                        batch = samples[offset : offset + batch_size]
                        self.assertEqual(
                            Counter(sample.mode for sample in batch),
                            {mode: batch_size // 8 for mode in range(8)},
                        )
                        members = {sample.index for sample in batch}
                        for index in members:
                            if epoch == 0:
                                companions[index] = members - {index}
                            else:
                                companions[index].intersection_update(members)
                self.assertTrue(all(not peers for peers in companions.values()))

    def test_every_glyph_covers_modes_and_visits_skip_rotating_omissions(self):
        visits = Counter()
        omitted = Counter()
        for cycle in range(32):
            modes = defaultdict(list)
            for phase in range(8):
                samples = epoch_samples(25, 8, 42, cycle * 8 + phase)
                for sample in samples:
                    self.assertEqual(sample.visit, visits[sample.index])
                    modes[sample.index].append(sample.mode)
            self.assertEqual(len(modes), 24)
            for index, seen in modes.items():
                self.assertEqual(sorted(seen), list(range(8)))
                visits[index] += 1
            for index in set(range(25)) - modes.keys():
                omitted[index] += 1
        self.assertEqual(set(omitted), set(range(25)))
        self.assertLessEqual(max(omitted.values()) - min(omitted.values()), 1)

    def test_large_remainder_and_no_remainder_both_preserve_coverage(self):
        for size, batch in ((15, 8), (32, 16), (39, 16)):
            visits = Counter()
            for cycle in range(16):
                modes = defaultdict(list)
                for phase in range(8):
                    samples = epoch_samples(size, batch, 7, cycle * 8 + phase)
                    self.assertEqual(len(samples), size // batch * batch)
                    for sample in samples:
                        self.assertEqual(sample.visit, visits[sample.index])
                        modes[sample.index].append(sample.mode)
                for index, seen in modes.items():
                    self.assertEqual(sorted(seen), list(range(8)))
                    visits[index] += 1

    def test_embedding_decks_cover_enabled_variants_without_replacement(self):
        for mode in range(8):
            gpt = range(4) if mode & 4 else range(1)
            clip = range(3) if mode & 2 else range(1)
            expected = set(itertools.product(gpt, clip))
            length = len(expected)
            for cycle in range(3):
                pairs = [
                    embedding_variants(ConditionSample(12, mode, visit), 42, 4, 3)
                    for visit in range(cycle * length, (cycle + 1) * length)
                ]
                self.assertEqual(set(pairs), expected)
                self.assertEqual(len(pairs), len(set(pairs)))

    def test_schedule_does_not_consume_global_rng_and_is_reproducible(self):
        before = torch.get_rng_state()
        samples = epoch_samples(129, 16, 42, 39)
        variants = [embedding_variants(s, 42, 4, 2) for s in samples]
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(samples, epoch_samples(129, 16, 42, 39))
        self.assertEqual(variants, [embedding_variants(s, 42, 4, 2) for s in samples])
        self.assertNotEqual(samples, epoch_samples(129, 16, 43, 39))

    def test_invalid_batch_and_dataset_fail(self):
        for size, batch, epoch in ((128, 127, 0), (7, 8, 0), (128, 128, -1)):
            with self.assertRaises(ValueError):
                epoch_samples(size, batch, 42, epoch)


if __name__ == "__main__":
    unittest.main()
