import unittest
from unittest.mock import patch

from sglang.srt.utils.common import set_gpu_proc_affinity


class FakeProcess:
    def __init__(self, allowed):
        self.allowed = list(allowed)
        self.bound = None

    def cpu_affinity(self, cpus=None):
        if cpus is not None:
            self.bound = list(cpus)
        return self.bound if self.bound is not None else self.allowed


class TestGpuProcAffinity(unittest.TestCase):
    def bind(self, allowed, gpu_id, tp_size=8):
        process = FakeProcess(allowed)
        with patch("sglang.srt.utils.common.psutil.Process", return_value=process):
            set_gpu_proc_affinity(1, tp_size, 1, gpu_id)
        return process.bound

    def test_contiguous_cpuset_partitions_tp8(self):
        partitions = [self.bind(range(128), rank) for rank in range(8)]
        self.assertEqual(partitions, [list(range(i, i + 16)) for i in range(0, 128, 16)])
        self.assertEqual(sorted(cpu for part in partitions for cpu in part), list(range(128)))

    def test_sparse_cpuset_uses_only_allowed_cpu_ids(self):
        allowed = [96, 2, 42, 8, 75, 19, 81, 33, 64]
        partitions = [self.bind(allowed, rank, tp_size=4) for rank in range(4)]
        self.assertEqual(partitions, [[2, 8, 19], [33, 42], [64, 75], [81, 96]])
        self.assertEqual(self.bind(allowed, 4, tp_size=4), partitions[0])

    def test_insufficient_allowed_cpus_fails_before_binding(self):
        process = FakeProcess([4, 9])
        with patch("sglang.srt.utils.common.psutil.Process", return_value=process):
            with self.assertRaisesRegex(ValueError, "Cannot bind 4 TP ranks"):
                set_gpu_proc_affinity(1, 4, 1, 0)
        self.assertIsNone(process.bound)


if __name__ == "__main__":
    unittest.main()
