import unittest
from unittest.mock import patch

from sglang.srt.utils.common import set_gpu_proc_affinity
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeProcess:
    def __init__(self, allowed):
        self.allowed = list(allowed)
        self.bound = None

    def cpu_affinity(self, cpus=None):
        if cpus is not None:
            self.bound = list(cpus)
        return self.bound if self.bound is not None else self.allowed


class TestGpuProcAffinity(unittest.TestCase):
    def bind(self, allowed, gpu_id, tp_rank, tp_size=8):
        process = FakeProcess(allowed)
        with patch("sglang.srt.utils.common.psutil.Process", return_value=process):
            set_gpu_proc_affinity(1, tp_size, 1, gpu_id, tp_rank)
        return process.bound

    def test_contiguous_cpuset_partitions_tp8(self):
        partitions = [self.bind(range(128), rank, rank) for rank in range(8)]
        self.assertEqual(
            partitions, [list(range(i, i + 16)) for i in range(0, 128, 16)]
        )
        self.assertEqual(
            sorted(cpu for part in partitions for cpu in part), list(range(128))
        )

    def test_sparse_cpuset_uses_only_allowed_cpu_ids(self):
        allowed = [96, 2, 42, 8, 75, 19, 81, 33, 64]
        partitions = [self.bind(allowed, rank, rank, tp_size=4) for rank in range(4)]
        self.assertEqual(partitions, [[2, 8, 19], [33, 42], [64, 75], [81, 96]])
        self.assertEqual(self.bind(allowed, 4, 4, tp_size=4), partitions[0])

    def test_gpu_id_step_does_not_skip_tp_partitions(self):
        allowed = list(range(8))
        partitions = [
            self.bind(allowed, rank * 2, rank, tp_size=4) for rank in range(4)
        ]
        self.assertEqual(partitions, [[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_reindexed_device_zero_still_uses_distinct_tp_partitions(self):
        allowed = list(range(8))
        partitions = [self.bind(allowed, 0, rank, tp_size=4) for rank in range(4)]
        self.assertEqual(partitions, [[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_insufficient_allowed_cpus_fails_before_binding(self):
        process = FakeProcess([4, 9])
        with patch("sglang.srt.utils.common.psutil.Process", return_value=process):
            with self.assertRaisesRegex(ValueError, "Cannot bind 4 TP ranks"):
                set_gpu_proc_affinity(1, 4, 1, 0, 0)
        self.assertIsNone(process.bound)

    def test_invalid_topology_rejected_before_affinity_read(self):
        process = FakeProcess([2, 4, 6])
        with patch("sglang.srt.utils.common.psutil.Process", return_value=process):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                set_gpu_proc_affinity(0, 4, 1, 0, 0)
        self.assertIsNone(process.bound)

    def test_readback_mismatch_is_reported(self):
        class DriftingProcess(FakeProcess):
            def cpu_affinity(self, cpus=None):
                if cpus is not None:
                    self.bound = list(cpus)
                    return None
                return self.allowed if self.bound is None else self.bound + [99]

        process = DriftingProcess([2, 4, 6, 8])
        with patch("sglang.srt.utils.common.psutil.Process", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "CPU affinity changed"):
                set_gpu_proc_affinity(1, 2, 1, 0, 0)

    def test_different_worker_cpusets_only_guarantee_local_membership(self):
        first = self.bind([0, 1, 2, 3], 0, 0, tp_size=2)
        second = self.bind([0, 1], 1, 1, tp_size=2)
        self.assertEqual(first, [0, 1])
        self.assertEqual(second, [1])


if __name__ == "__main__":
    unittest.main()
