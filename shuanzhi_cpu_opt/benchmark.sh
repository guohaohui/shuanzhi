#!/bin/bash
# benchmark.sh — CPU optimization benchmark (legacy standalone)
set -e
THREADS=${OMP_NUM_THREADS:-4}
export OMP_NUM_THREADS=$THREADS

echo "CPU Opt Benchmark (OMP=$THREADS)"

for algo in rk4 kalman hungarian ekf; do
  for b in ${algo}_baseline ${algo}_omp ${algo}_opt; do
    [ -x "$b" ] && { echo "--- $b ---"; ./"$b"; }
  done
done
