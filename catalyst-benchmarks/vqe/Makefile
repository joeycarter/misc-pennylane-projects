.PHONY: benchmarks help plots

# Help message
help:
	@echo "Usage:"
	@echo "  make benchmarks    Run the VQE benchmark with pytest"
	@echo "  make plots         Plot the VQE benchmarks"

# Run benchmarks
benchmarks:
	mkdir -p benchmarks
	python3 -m pytest benchmarks/ -s -vv -m fast --benchmark-enable --benchmark-warmup=off --benchmark-min-rounds=10 --benchmark-json=vqe_benchmarks.json vqe_benchmark.py

# Make plots
plots:
	python3 plot.py vqe_benchmark.py -v