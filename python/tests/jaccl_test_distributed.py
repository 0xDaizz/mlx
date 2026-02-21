# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx_distributed_tests
import mlx_tests


class TestJACCLDistributed(mlx_distributed_tests.MLXDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        _ = mx.distributed.init(strict=True, backend="jaccl")
        cls.atol = 1e-6
        cls.rtol = 1e-4

    def test_groups(self):
        world = mx.distributed.init()
        self.assertEqual(world.size(), 2)
        self.assertTrue(0 <= world.rank() < 2)

        world2 = mx.distributed.init()
        self.assertEqual(world.size(), world2.size())
        self.assertEqual(world.rank(), world2.rank())


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
