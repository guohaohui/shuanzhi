#!/bin/bash
# ft2000_bench.sh — FT2000 (ARM64) benchmark script
set -e

THREADS=${OMP_NUM_THREADS:-8}
export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static

ARCH=${1:-sm_61}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT="ft2000_bench_result_${TIMESTAMP}.txt"

echo "shuanzhi FT2000 Benchmark OMP=$THREADS ARCH=$ARCH" | tee "$RESULT"

mkdir -p cpu_base cpu_omp cpu_simd cpu_float cpu_simd_float cpu_omp_simd cpu_omp_simd_float cpu_opt
mkdir -p gpu_base gpu_opt

TESTSIZE=65536
BATCHSIZE=65536
MATSIZE=8

# CPU compilation (ARM64: no -mavx2/-mfma)
for algo in rk4 kalman hungarian ekf; do
  SRC="cpu_uopt_src/${algo}_uopt.c"
  gcc -O2 $SRC -lm -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_base/${algo}_base 2>&1 | tail -1
  gcc -O2 -DOPT_OPENMP $SRC -lm -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_omp/${algo}_omp 2>&1 | tail -1
  gcc -O2 -DOPT_SIMD $SRC -lm -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_simd/${algo}_simd 2>&1 | tail -1
  gcc -O2 -DOPT_FLOAT $SRC -lm -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_float/${algo}_float 2>&1 | tail -1
  gcc -O2 -DOPT_SIMD -DOPT_FLOAT $SRC -lm -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_simd_float/${algo}_simd_float 2>&1 | tail -1
  gcc -O2 -DOPT_OPENMP -DOPT_SIMD $SRC -lm -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_omp_simd/${algo}_omp_simd 2>&1 | tail -1
  gcc -O2 -DOPT_OPENMP -DOPT_SIMD -DOPT_FLOAT $SRC -lm -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_omp_simd_float/${algo}_omp_simd_float 2>&1 | tail -1
  gcc -O2 -DOPT_OPENMP -DOPT_SIMD -DOPT_FLOAT -DOPT_ALGO $SRC -lm -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_opt/${algo}_opt 2>&1 | tail -1
done

# GPU compilation (sm_61 for P400)
nvcc -arch=$ARCH -O2 gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -o gpu_base/rk4_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 -DOPT_ALGO -DOPT_MEM gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -o gpu_opt/rk4_gpu_opt 2>&1 | tail -1
nvcc -arch=$ARCH -O2 gpu_uopt_src/kalman_gpu_uopt.cu -DBATCH_SIZE=$BATCHSIZE -o gpu_base/kalman_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 -DOPT_ALGO -DOPT_MEM gpu_uopt_src/kalman_gpu_uopt.cu -DBATCH_SIZE=$BATCHSIZE -o gpu_opt/kalman_gpu_opt 2>&1 | tail -1
nvcc -arch=$ARCH -O2 gpu_uopt_src/hungarian_gpu_uopt.cu -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_base/hungarian_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_ALGO gpu_uopt_src/hungarian_gpu_uopt.cu -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_opt/hungarian_gpu_opt 2>&1 | tail -1
nvcc -arch=$ARCH -O2 gpu_uopt_src/ekf_gpu_uopt.cu -DBATCH=$BATCHSIZE -o gpu_base/ekf_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 -DOPT_ALGO gpu_uopt_src/ekf_gpu_uopt.cu -DBATCH=$BATCHSIZE -o gpu_opt/ekf_gpu_opt 2>&1 | tail -1

# Run all tests
for dir in cpu_base cpu_omp cpu_simd cpu_float cpu_simd_float cpu_omp_simd cpu_omp_simd_float cpu_opt; do
  for b in $dir/*; do [ -x "$b" ] && { echo "--- $b ---" | tee -a "$RESULT"; ./"$b" 2>&1 | tee -a "$RESULT"; } done
done
for dir in gpu_base gpu_opt; do
  for b in $dir/*; do [ -x "$b" ] && { echo "--- $b ---" | tee -a "$RESULT"; ./"$b" 2>&1 | tee -a "$RESULT"; } done
done

echo "Done: $RESULT"
