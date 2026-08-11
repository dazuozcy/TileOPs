"""Workload definitions for the reduction op family."""

from npu_bench.workloads.workload_base import RandnWorkload


class LogSumExpWorkload(RandnWorkload):
    """Workload definition for LogSumExpFwdOp (spec interface: shape + dtype)."""
