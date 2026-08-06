.PHONY: test bench bench-smoke bench-msprof help

test:
	python -m pytest -q tests

bench:
	python -m pytest benchmarks/

bench-smoke:
	python -m pytest benchmarks/ -m smoke

bench-msprof:
	NPU_BENCH_TIMING=msprof python -m pytest benchmarks/

help:
	@echo "Available targets:"
	@echo "  test         Run correctness tests"
	@echo "  bench        Run all benchmarks (event timing)"
	@echo "  bench-smoke  Run smoke-tier benchmarks only"
	@echo "  bench-msprof Run all benchmarks (msprof op timing)"
	@echo ""
	@echo "Environment variables:"
	@echo "  NPU_BENCH_TIMING=msprof              Use msprof op profiling"
	@echo "  NPU_BENCH_FORCE_CUDA=1               Use CUDA backend (for local dev)"
	@echo "  NPU_BENCH_FLUSH_CACHE=0              Disable L2 cache flush"
	@echo "  NPU_BENCH_MSPROF_LAUNCH_COUNT=10     msprof timed launch count"
	@echo "  NPU_BENCH_MSPROF_WARM_UP=5           msprof warm-up launch count"
	@echo "  NPU_BENCH_MSPROF_KERNEL_NAME=main    Override kernel name"
	@echo "  NPU_BENCH_MSPROF_OUTPUT=/path        Fixed msprof output directory"
