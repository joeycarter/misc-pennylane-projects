.PHONY: benchmarks help plots

# Help message
help:
	@echo "Usage:"
	@echo "  make benchmarks    Run the XAS benchmark with pytest"
	@echo "  make plots         Plot the XAS benchmarks"

# Run benchmarks
benchmarks:
	mkdir -p benchmarks
	python3 -m pytest benchmarks/ -s -vv -m fast --benchmark-enable --benchmark-warmup=off --benchmark-min-rounds=10 --benchmark-json=xas_benchmarks.json xas_benchmark.py

# Make plots
plots:
	python3 plot.py xas_benchmarks.json -v