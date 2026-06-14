# shuanzhi — 高性能数值算法优化项目

四个经典数值算法的 CPU 多线程+SIMD+GPU 完整优化实现，展示从"原版串行"到"全优化"的完整性能调优路径。

**支持双平台：** x86_64 (AVX2+FMA) ⇄ ARM64 (NEON) · **50版本全通过 · 152断言 PASS · 0 FAIL**

---

## 算法概览

| 算法 | 问题类型 | 数学核心 | 规模 |
|------|---------|---------|------|
| **RK4 / RK4SD** | 常微分方程初值问题 | 4阶龙格-库塔 + GSL步长折半误差估计 | 65536点 × 100步 |
| **卡尔曼滤波** | 线性系统状态估计 | 7×7矩阵乘法 + 高斯消元/Cholesky | 65536问题 × 100步 |
| **匈牙利算法** | 组合优化（指派问题） | Munkres O(n³) + 位掩码加速 | 65536个 8×8矩阵 |
| **扩展卡尔曼滤波** | 非线性状态估计+参数辨识 | 4D阻尼摆 + 雅可比线性化 + 解析2×2逆 | 65536问题 × 100步 |

---

## 优化体系

### CPU 优化（8级，3个正交维度 + 1个算法维度）

| 宏 | 优化 | x86_64 (AVX2) | ARM64 (NEON) |
|------|------|------|------|
| `OPT_OPENMP` | 多线程并行 (4/8核) | 3.3-4.2× | 2.6-5.0× |
| `OPT_SIMD` | SIMD向量化 | 256-bit, 4×double / 8×float | 128-bit, 2×double / 4×float |
| `OPT_FLOAT` | 精度降级 double→float | 配合SIMD翻倍吞吐 | 配合NEON翻倍吞吐 |
| `OPT_ALGO` | 算法优化 | 分块/展开/Cholesky/位掩码/解析逆 | 同左 |

### GPU 优化（5级，3个正交维度）

| 宏 | 优化 | 说明 |
|------|------|------|
| `OPT_FP32` | FP64→FP32 | GPU FP32吞吐是FP64的2-64倍 |
| `OPT_ALGO` | 算法优化 | 预计算常量/矩阵展开/Cholesky/位掩码/解析逆 |
| `OPT_MEM` | 内存层级 | Shared Memory (Kalman) / Constant Memory (EKF) |

---

## 项目结构

```
├── cpu_uopt_src/          CPU 统一优化源文件 (8级, 编译宏切换)
├── gpu_uopt_src/          GPU 统一优化源文件 (5级, 编译宏切换)
├── shuanzhi/              基线 demo (CPU/GPU 双模式)
├── shuanzhi_cpu_opt/      旧版 CPU 独立优化 (保留参考)
├── shuanzhi_gpu_opt/      旧版 GPU 独立优化 (保留参考)
├── bench_all.sh           x86_64 一键编译+测试脚本
├── ft2000_bench.sh        FT2000(ARM64) 编译+测试脚本
├── shuanzhi-完整技术文档.md  项目核心技术文档
├── GPU优化详解.md          GPU 优化方法详解
├── 优化维度详解.md          优化标志与代码改动对照
└── 版本-二进制-编译命令对照表.md
```

---

## 快速开始

### x86_64 开发环境

```bash
# 一键编译+测试 (需要 GCC + NVCC)
OMP_NUM_THREADS=4 ./bench_all.sh

# 或手动编译单个版本
gcc -O2 -DOPT_OPENMP -DOPT_SIMD -DOPT_FLOAT -DOPT_ALGO \
    cpu_uopt_src/rk4_uopt.c -lm -mavx2 -mfma -fopenmp \
    -DTEST_SIZE=65536 -o rk4_opt
./rk4_opt
```

### FT2000 (ARM64) 生产环境

```bash
# sm_61 适配 Quadro P400
OMP_NUM_THREADS=8 bash ft2000_bench.sh

# 手动 GPU 编译
nvcc -arch=sm_61 -O2 -DOPT_FP32 -DOPT_ALGO \
     gpu_uopt_src/rk4_gpu_uopt.cu -DTEST_SIZE=65536 -o rk4_gpu_opt
```

---

## 性能一览 (x86_64, RTX 4060 Ti)

| 算法 | CPU Baseline | CPU Full Opt | GPU OPT | 最大加速比 |
|------|------|------|------|------|
| **RK4** | 0.0417s | 0.0013s (32×) | 0.0004s | **104×** |
| **Kalman** | 6.2104s | 0.9506s (6.5×) | 0.0080s | **777×** |
| **Hungarian** | 0.1290s | 0.0191s (6.8×) | 0.0090s | **45×** |
| **EKF** | 1.0800s | 0.1571s (6.9×) | 0.0006s | **267×** |

---

## 平台兼容

| | 开发环境 | 生产环境 |
|------|------|------|
| **CPU** | Intel i5-14600KF (x86_64, AVX2+FMA) | Phytium FT2000/64 (ARMv8-A, NEON) |
| **GPU** | NVIDIA RTX 4060 Ti (sm_89) | NVIDIA Quadro P400 (sm_61, 2GB) |
| **OS** | Ubuntu (WSL2) | Kylin V10 (aarch64) |
| **编译器** | GCC 9.4 + NVCC 12.6 | GCC 7.x + NVCC 11.x |

通过 `#ifdef __aarch64__` / `#ifdef __x86_64__` 条件编译保证双平台兼容。

---

## 参考库

| 算法 | 参考库 | 验证方式 |
|------|------|---------|
| RK4 | GNU GSL (rk4.c) | 操作序列位级复现 |
| Kalman | OpenCV (test_kalman.cpp) | 7维100步标准配置 |
| Hungarian | Google OR-Tools | 4×4标准矩阵 (min=275) + 暴力枚举 n=3~6 |
| EKF | 自实现 | 雅可比解析 vs 数值微分验证 |

所有算法**参考而不依赖**——零外部库依赖，纯 C/CUDA 实现。
