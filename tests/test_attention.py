import math
import unittest

import torch

from mojika.conditional.modules.attention import Attention
from mojika.utils.init_positions_2d import init_positions_2d


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_signed_per_head_gains_match_l2_attention_and_gradients(self):
        for head_dim in (8, 64):
            with self.subTest(head_dim=head_dim):
                dim = 3 * head_dim
                attention = Attention(dim, num_heads=3, qk_norm=True, rope_theta=10)
                self.assertEqual(attention.q_scale.shape, (1, 1, 3, head_dim))
                torch.testing.assert_close(
                    attention.q_scale, torch.ones_like(attention.q_scale)
                )
                with torch.no_grad():
                    attention.q_scale.copy_(
                        torch.linspace(-1.5, 1.5, dim).reshape(1, 1, 3, head_dim)
                    )
                    attention.q_scale[0, 0, 0, 0] = 0
                x = torch.randn(2, 6, dim, requires_grad=True)
                positions = init_positions_2d(3, 2)
                actual = attention(x, positions)

                q, k, v = attention.qkv(x).reshape(2, 6, 3, 3, head_dim).unbind(2)
                # RMSNorm + 1/sqrt(d) equals L2 normalization + sqrt(d),
                # including the corresponding epsilon in the L2 denominator.
                q = q / (q.square().sum(-1, keepdim=True) + head_dim * 1e-6).sqrt()
                k = k / (k.square().sum(-1, keepdim=True) + head_dim * 1e-6).sqrt()
                q, k = attention.rope(q * attention.q_scale, k, positions)
                logits = torch.einsum("bnhd,bmhd->bhnm", q, k) * math.sqrt(head_dim)
                values = torch.einsum("bhnm,bmhd->bnhd", logits.softmax(-1), v)
                expected = attention.proj(values.reshape(2, 6, dim))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

                upstream = torch.randn_like(actual)
                parameters = (x, attention.q_scale, attention.qkv.weight)
                actual_grads = torch.autograd.grad(actual, parameters, upstream)
                expected_grads = torch.autograd.grad(expected, parameters, upstream)
                for actual_grad, expected_grad in zip(
                    actual_grads, expected_grads, strict=True
                ):
                    self.assertTrue(torch.isfinite(actual_grad).all())
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=2e-5, atol=2e-6
                    )
                self.assertTrue((actual_grads[1] != 0).all())

    def test_disabled_qk_norm_matches_unnormalized_attention(self):
        attention = Attention(24, num_heads=3, qk_norm=False, rope_theta=10)
        self.assertIsNone(attention.q_scale)
        x = torch.randn(2, 6, 24)
        positions = init_positions_2d(3, 2)
        q, k, v = attention.qkv(x).reshape(2, 6, 3, 3, 8).unbind(2)
        q, k = attention.rope(q, k, positions)
        logits = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(8)
        values = torch.einsum("bhnm,bmhd->bnhd", logits.softmax(-1), v)
        expected = attention.proj(values.reshape(2, 6, 24))
        torch.testing.assert_close(attention(x, positions), expected)



if __name__ == "__main__":
    unittest.main()
