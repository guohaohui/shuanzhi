#!/bin/bash
# bench_all.sh — x86_64 一键测试脚本
# 用法: OMP_NUM_THREADS=4 ./bench_all.sh
# 输出: bench_result_<timestamp>.txt + bench_summary_<timestamp>.txt

set -e

THREADS=${OMP_NUM_THREADS:-4}
export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static

ARCH=${1:-sm_89}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT="bench_result_${TIMESTAMP}.txt"
SUMMARY="bench_summary_${TIMESTAMP}.txt"

echo "========================================" | tee "$RESULT"
echo "  shuanzhi Benchmark Suite (x86_64)" | tee -a "$RESULT"
echo "  OMP_NUM_THREADS=$THREADS  ARCH=$ARCH" | tee -a "$RESULT"
echo "  Timestamp: $TIMESTAMP" | tee -a "$RESULT"
echo "========================================" | tee -a "$RESULT"

# ========== 创建输出目录 ==========
mkdir -p cpu_base cpu_omp cpu_simd cpu_float cpu_simd_float cpu_omp_simd cpu_omp_simd_float cpu_opt
mkdir -p gpu_base gpu_fp32 gpu_algo gpu_mem gpu_opt

TESTSIZE=65536
BATCHSIZE=65536
MATSIZE=8

# ================================================================
# 1. CPU 编译
# ================================================================
echo "" | tee -a "$RESULT"
echo "===== CPU 编译 =====" | tee -a "$RESULT"

for algo in rk4 kalman hungarian ekf; do
  SRC="cpu_uopt_src/${algo}_uopt.c"
  
  # Baseline
  gcc -O2 $SRC -lm -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_base/${algo}_base 2>&1 | tail -1
  
  # OMP
  gcc -O2 -DOPT_OPENMP $SRC -lm -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_omp/${algo}_omp 2>&1 | tail -1
  
  # SIMD (double)
  gcc -O2 -DOPT_SIMD $SRC -lm -mavx2 -mfma -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_simd/${algo}_simd 2>&1 | tail -1
  
  # Float
  gcc -O2 -DOPT_FLOAT $SRC -lm -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_float/${algo}_float 2>&1 | tail -1
  
  # SIMD + Float
  gcc -O2 -DOPT_SIMD -DOPT_FLOAT $SRC -lm -mavx2 -mfma -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_simd_float/${algo}_simd_float 2>&1 | tail -1
  
  # OMP + SIMD
  gcc -O2 -DOPT_OPENMP -DOPT_SIMD $SRC -lm -mavx2 -mfma -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_omp_simd/${algo}_omp_simd 2>&1 | tail -1
  
  # OMP + SIMD + Float
  gcc -O2 -DOPT_OPENMP -DOPT_SIMD -DOPT_FLOAT $SRC -lm -mavx2 -mfma -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_omp_simd_float/${algo}_omp_simd_float 2>&1 | tail -1
  
  # Full Opt
  gcc -O2 -DOPT_OPENMP -DOPT_SIMD -DOPT_FLOAT -DOPT_ALGO $SRC -lm -mavx2 -mfma -fopenmp -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o cpu_opt/${algo}_opt 2>&1 | tail -1
done

echo "CPU 编译完成" | tee -a "$RESULT"

# ================================================================
# 2. GPU 编译
# ================================================================
echo "" | tee -a "$RESULT"
echo "===== GPU 编译 =====" | tee -a "$RESULT"

# RK4 GPU
nvcc -arch=$ARCH -O2 gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_base/rk4_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_fp32/rk4_gpu_fp32 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_ALGO gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_algo/rk4_gpu_algo 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_MEM gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_mem/rk4_gpu_mem 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 -DOPT_ALGO -DOPT_MEM gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_opt/rk4_gpu_opt 2>&1 | tail -1

# Kalman GPU
nvcc -arch=$ARCH -O2 gpu_uopt_src/kalman_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_base/kalman_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 gpu_uopt_src/kalman_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_fp32/kalman_gpu_fp32 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_ALGO gpu_uopt_src/kalman_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_algo/kalman_gpu_algo 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_MEM gpu_uopt_src/kalman_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_mem/kalman_gpu_mem 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 -DOPT_ALGO -DOPT_MEM gpu_uopt_src/kalman_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_opt/kalman_gpu_opt 2>&1 | tail -1

# Hungarian GPU
nvcc -arch=$ARCH -O2 gpu_uopt_src/hungarian_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_base/hungarian_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 gpu_uopt_src/hungarian_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_fp32/hungarian_gpu_fp32 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_ALGO gpu_uopt_src/hungarian_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_algo/hungarian_gpu_algo 2>&1 | tail -1
# GPU OPT = ALGO (FP32+ALGO has precision issues with bitmask)
nvcc -arch=$ARCH -O2 -DOPT_ALGO gpu_uopt_src/hungarian_gpu_uopt.cu -DTEST_SIZE=$TESTSIZE -DBATCH_SIZE=$BATCHSIZE -DMATRIX_SIZE=$MATSIZE -o gpu_opt/hungarian_gpu_opt 2>&1 | tail -1

# EKF GPU
nvcc -arch=$ARCH -O2 gpu_uopt_src/ekf_gpu_uopt.cu -DBATCH=$BATCHSIZE -o gpu_base/ekf_gpu_base 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 gpu_uopt_src/ekf_gpu_uopt.cu -DBATCH=$BATCHSIZE -o gpu_fp32/ekf_gpu_fp32 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_ALGO gpu_uopt_src/ekf_gpu_uopt.cu -DBATCH=$BATCHSIZE -o gpu_algo/ekf_gpu_algo 2>&1 | tail -1
nvcc -arch=$ARCH -O2 -DOPT_FP32 -DOPT_ALGO gpu_uopt_src/ekf_gpu_uopt.cu -DBATCH=$BATCHSIZE -o gpu_opt/ekf_gpu_opt 2>&1 | tail -1

echo "GPU 编译完成" | tee -a "$RESULT"

# ================================================================
# 3. CPU 测试
# ================================================================
echo "" | tee -a "$RESULT"
echo "===== CPU 测试 =====" | tee -a "$RESULT"

CPU_DIRS="cpu_base cpu_omp cpu_simd cpu_float cpu_simd_float cpu_omp_simd cpu_omp_simd_float cpu_opt"
CPU_ALGOS="rk4 kalman hungarian ekf"

for dir in $CPU_DIRS; do
  for algo in $CPU_ALGOS; do
    bin="${dir}/${algo}_*"
    for b in $bin; do
      if [ -x "$b" ]; then
        echo "" | tee -a "$RESULT"
        echo "--- Running: $b ---" | tee -a "$RESULT"
        ./"$b" 2>&1 | tee -a "$RESULT"
      fi
    done
  done
done

# ================================================================
# 4. GPU 测试
# ================================================================
echo "" | tee -a "$RESULT"
echo "===== GPU 测试 =====" | tee -a "$RESULT"

GPU_DIRS="gpu_base gpu_fp32 gpu_algo gpu_mem gpu_opt"
GPU_ALGOS="rk4 kalman hungarian ekf"

for dir in $GPU_DIRS; do
  for algo in $GPU_ALGOS; do
    bin="${dir}/${algo}_gpu_*"
    for b in $bin; do
      if [ -x "$b" ]; then
        echo "" | tee -a "$RESULT"
        echo "--- Running: $b ---" | tee -a "$RESULT"
        ./"$b" 2>&1 | tee -a "$RESULT"
      fi
    done
  done
done

# ================================================================
# 5. 生成汇总
# ================================================================
echo "" | tee -a "$RESULT"
echo "===== 汇总 =====" | tee -a "$RESULT"

python3 - <<PYEOF | tee "$SUMMARY"
import re, sys

with open('$RESULT') as f:
    text = f.read()

# Extract PASS/FAIL counts
pass_count = len(re.findall(r'ALL PASS|PASS', text))
fail_count = len(re.findall(r'\bFAIL\b|errors', text))

# Extract timing info (simple heuristic)
timings = re.findall(r'(CPU|GPU)\s*(?:耗时|时间|time).*?([\d.]+)\s*s', text, re.I)

print(f"Test Summary")
print(f"============")
print(f"PASS assertions: {pass_count}")
print(f"FAIL assertions: {fail_count}")
print(f"Timing entries found: {len(timings)}")
PYEOF

echo "" | tee -a "$RESULT"
echo "========================================" | tee -a "$RESULT"
echo "  测试完成" | tee -a "$RESULT"
echo "  结果: $RESULT" | tee -a "$RESULT"
echo "  汇总: $SUMMARY" | tee -a "$RESULT"
echo "========================================" | tee -a "$RESULT"
