"""Run COMPASS on the separately registered mechanism benchmarks."""

from experiments.paper.run_compass_reflection import main


if __name__ == "__main__":
    raise SystemExit(main(benchmark_family="mechanism"))
