# shuanzhi 数值算法项目 — 完整技术文档

**版本:** 2.3 | **日期:** 2026-06-02
**平台:** i5-14600KF + RTX 4060 Ti (x86_64) / FT2000 + P400 (ARM64)

---

## 第一章  项目概述

### 1.1  项目背景与目标

本项目实现并优化了四个经典数值算法，支持 CPU 和 GPU 双模式运行。目标是在统一代码框架下，展示从"原版串行实现"到"CPU 多线程+SIMD 优化"再到"GPU kernel 优化"的完整性能调优路径。

四个算法覆盖了数值计算中最常见的三类问题：

| 算法 | 问题类型 | 数学核心 |
|------|---------|---------|
| RK4 / RK4SD | 常微分方程 (ODE) 初值问题 | 4阶龙格-库塔 / GSL 步长折半 |
| 卡尔曼滤波 | 线性系统状态估计 | 矩阵乘法 + 高斯消元 |
| 匈牙利算法 | 组合优化 (指派问题) | Munkres (O(n³)) + 增广路径 |
| 扩展卡尔曼滤波 (EKF) | 非线性系统状态估计 + 参数辨识 | 雅可比线性化 + 矩阵求逆 |

### 1.2  四个算法简介

**RK4 (龙格-库塔 4 阶):** 求解 ODE `dy/dt = -y`，已知解析解 `y = y₀·e^{-t}`，可精确验证。每个初始点独立积分，65536 个点天然适合并行。

**RK4SD (GSL 步长折半):** 同一 ODE，但采用 GSL 的 Step-Doubling 方法——一整步 h + 两个半步 h/2，用 12 次函数求值（经典 RK4 的 3 倍）换取更高精度和误差估计能力。

**卡尔曼滤波 (Kalman Filter):** 7 维线性系统状态估计。F=H=Q=I, R=σ²I。65536 个独立滤波问题批量处理。

**匈牙利算法 (Hungarian Algorithm / Munkres):** 求解 n×n 指派问题，最小化总成本。n=8, 65536 个随机成本矩阵批量处理。用 n=3~6 暴力枚举验证正确性。

**扩展卡尔曼滤波 (EKF):** 4D 阻尼摆 `[θ, ω, α, β]` 的状态估计和参数在线辨识（真值 α=9.81, β=0.30）。非线性观测模型 `[θ, sin(θ)]`。

### 1.3  优化层级简介

项目按"正交维度"组织优化，通过编译宏 `-D` 组合出不同版本。CPU 有 8 级（3 个正交维度 + 1 个算法维度），GPU 有 5 级（精度/算法/内存三个维度，按算法不同组合）。

**统一源文件：** CPU 版本共用 `cpu_uopt_src/{rk4,kalman,hungarian,ekf}_uopt.c`，GPU 版本共用 `gpu_uopt_src/{rk4,kalman,hungarian,ekf}_gpu_uopt.cu`（EKF 另有旧版 `ekf_gpu_base.cu` 保留参考）。详见 2.5 节。

### 1.4  测试平台

| | x86_64 | ARM64 (FT2000) |
|------|------|------|
| **CPU** | Intel Core i5-14600KF (14核20线程, Raptor Lake, 3.5 GHz) | Phytium FT2000/64 (64核, ARMv8-A) |
| **内存** | 15 GB DDR5 | — |
| **GPU** | NVIDIA RTX 4060 Ti (8GB, 4352核, Ada Lovelace, sm_89) | NVIDIA Quadro P400 (2GB, 256核, Pascal, sm_61) |
| **OS** | Ubuntu 20.04 (WSL2) | Kylin V10 (aarch64) |
| **GCC** | 9.4.0 | 7.x+ |
| **NVCC** | 12.6 | 11.x (P400 不支持 12.x) |
| **SIMD** | AVX2 + FMA (256-bit) | ARM NEON (128-bit) |
| **线程数** | OMP_NUM_THREADS=4 | OMP_NUM_THREADS=8 (推荐) |
| **测试脚本** | `bench_all.sh` | `ft2000_bench.sh` |

### 1.5  文档说明

本文档是项目唯一的技术文档，涵盖项目概述、文件结构、优化体系、算法实现、性能测试等完整内容。

---

## 第二章  文件结构

### 2.1  完整目录树

```
test/
├── shuanzhi/                         基线 (CPU/GPU 双模式 .cu)
│   ├── include/common.h              公共头文件 (计时/验证/随机数)
│   ├── rk4_gsl_demo.cu               RK4 经典 + GSL步长折半
│   ├── kalman_demo.cu                卡尔曼滤波
│   ├── hungarian_demo.cu             匈牙利算法
│   └── ekf_demo.cu                   扩展卡尔曼滤波
│
├── cpu_uopt_src/                     CPU 统一优化源文件
│   ├── rk4_uopt.c                    RK4 (8级优化, 编译宏切换)
│   ├── kalman_uopt.c                 Kalman (8级优化)
│   ├── hungarian_uopt.c              Hungarian (8级优化)
│   └── ekf_uopt.c                    EKF (8级优化)
│
├── gpu_uopt_src/                     GPU 统一优化源文件
│   ├── rk4_gpu_uopt.cu               RK4 (FP32/ALGO/MEM 三维度)
│   ├── kalman_gpu_uopt.cu            Kalman (FP32/ALGO/MEM)
│   ├── hungarian_gpu_uopt.cu         Hungarian (FP32/ALGO)
│   ├── ekf_gpu_uopt.cu               EKF 统一源文件 (FP32/ALGO)
│   └── ekf_gpu_base.cu               EKF 基线 (旧版独立文件, 保留参考)
│
├── shuanzhi_cpu_opt/                 CPU 优化版 (旧版独立文件, 保留)
├── shuanzhi_gpu_opt/                 GPU 优化版 (旧版独立文件, 含4算法+include/, 保留)
│
├── bench_all.sh                      x86 一键测试脚本
├── ft2000_bench.sh                   FT2000 测试脚本
│
├── cpu_base/ cpu_omp/ cpu_simd/ cpu_float/     ← [生成] CPU 二进制 (8个目录)
├── cpu_simd_float/ cpu_omp_simd/ cpu_omp_simd_float/ cpu_opt/
├── gpu_base/ gpu_fp32/ gpu_algo/ gpu_mem/ gpu_opt/  ← [生成] GPU 二进制 (5个目录)
├── bench_result_*.txt / bench_summary_*.txt            ← [生成] 测试输出
├── 优化维度详解.md                    优化标志与代码改动对照文档
│
└── shuanzhi-完整技术文档.md           本文件
```

### 2.2  源代码目录说明

**`shuanzhi/` — 基线代码**
- `.cu` 扩展名文件通过 `#ifdef __CUDACC__` 宏实现 CPU/GPU 双模式编译
- CPU 编译: `gcc -DUSE_CPU -x c`，将 `__global__`/`__device__` 定义为空宏
- GPU 编译: `nvcc`，原生 CUDA kernel
- GPU 并行策略: "每个线程一个独立问题"——65536 个 RK4 点 → 65536 个 CUDA 线程

**`shuanzhi_cpu_opt/` — CPU 优化版**
- 每个算法两个文件: `*_omp.c` (仅加并行+SIMD, 算法不变) 和 `*_cpu_opt.c` (全优化)
- 纯 C 代码，仅需 GCC + OpenMP 编译
- ARM NEON / x86 AVX2 通过 `#if defined(__AVX2__)` / `#if HAS_NEON` 自动选择

**`shuanzhi_gpu_opt/` — GPU 优化版 (旧版, 保留参考)**
- 每个算法一个独立 `.cu` 文件 + `include/gpu_common.h`
- 当前测试脚本已改用 `gpu_uopt_src/` 下的统一源文件编译
- 此目录保留作为旧版独立实现的参考

### 2.3  编译产物目录

| 目录 | 内容 | 编译方式 |
|------|------|---------|
| `cpu_base/` | 4算法 基线 | GCC -O2 (无优化宏) |
| `cpu_omp/` | 4算法 OpenMP | GCC -O2 -DOPT_OPENMP -fopenmp |
| `cpu_simd/` | 4算法 SIMD(double) | GCC -O2 -DOPT_SIMD |
| `cpu_float/` | 4算法 Float | GCC -O2 -DOPT_FLOAT |
| `cpu_simd_float/` | 4算法 SIMD+Float | GCC -O2 -DOPT_SIMD -DOPT_FLOAT |
| `cpu_omp_simd/` | 4算法 OMP+SIMD | GCC -O2 -DOPT_OPENMP -DOPT_SIMD -fopenmp |
| `cpu_omp_simd_float/` | 4算法 OMP+SIMD+Float | GCC -O2 三者叠加 -fopenmp |
| `cpu_opt/` | 4算法 全优化 | GCC -O2 +全部宏 + -fopenmp |
| `gpu_base/` | GPU 基线 (double, 无优化宏) | NVCC -arch=sm_XX -O2 |
| `gpu_fp32/` | GPU FP32 (double→float) | NVCC -DOPT_FP32 |
| `gpu_algo/` | GPU 算法优化 | NVCC -DOPT_ALGO |
| `gpu_mem/` | GPU 内存优化 | NVCC -DOPT_MEM |
| `gpu_opt/` | GPU 全优化 | NVCC -DOPT_FP32 -DOPT_ALGO [-DOPT_MEM] |

### 2.4  测试脚本

| 脚本 | 平台 | 用法 |
|------|------|------|
| `bench_all.sh` | x86_64 | `./bench_all.sh` (默认 OMP=4, sm_89) |
| `ft2000_bench.sh` | ARM64 FT2000 | `bash ft2000_bench.sh` (默认 OMP=8, sm_61) |

两个脚本功能一致:
1. 编译 CPU 8 级 × 4 算法 + GPU 多级优化
2. 运行全部测试并采集完整输出
3. 生成 `bench_result_*.txt` (原始输出) 和 `bench_summary_*.txt` (Python 自动汇总)

### 2.5  优化层级总览

项目按"正交维度"组织优化，CPU 有 3 个独立维度（并行/向量化/精度）+ 1 个算法维度，GPU 有 3 个独立维度（精度/算法/内存）。每个维度可独立开关，通过编译宏 `-D` 组合出不同版本。

#### 2.5.1  编译宏定义

| 宏 | 作用 |
|------|------|
| `-DOPT_OPENMP` | OpenMP 多线程并行 |
| `-DOPT_SIMD` | SIMD 向量化（默认 double，配合 OPT_FLOAT 时用 float） |
| `-DOPT_FLOAT` | 精度缩减 double→float |
| `-DOPT_ALGO` | 算法级优化（分块/展开/Cholesky/位掩码/解析逆） |
| `-DOPT_MEM` | GPU 内存优化（Shared Memory / Constant Memory） |
| `-DOPT_FP32` | GPU 专用：FP64→FP32 |

#### 2.5.2  CPU 八级优化

**源文件：** `cpu_uopt_src/{rk4,kalman,hungarian,ekf}_uopt.c`（每个算法一个统一源文件）

| # | 目录 | 编译宏 | 含义 |
|---|------|------|------|
| 0 | `cpu_base/` | （无） | 基线，GCC -O2 串行 |
| 1 | `cpu_omp/` | `-DOPT_OPENMP` | 纯 OpenMP 多线程 |
| 2 | `cpu_simd/` | `-DOPT_SIMD` | 纯手写 SIMD（double） |
| 3 | `cpu_float/` | `-DOPT_FLOAT` | 纯精度缩减 double→float |
| 4 | `cpu_simd_float/` | `-DOPT_SIMD -DOPT_FLOAT` | SIMD + float |
| 5 | `cpu_omp_simd/` | `-DOPT_OPENMP -DOPT_SIMD` | OpenMP + SIMD（double） |
| 6 | `cpu_omp_simd_float/` | `-DOPT_OPENMP -DOPT_SIMD -DOPT_FLOAT` | 三者叠加 |
| 7 | `cpu_opt/` | `-DOPT_OPENMP -DOPT_SIMD -DOPT_FLOAT -DOPT_ALGO` | 全优化 |

**SIMD 精度规则：** SIMD 默认用 double（x86: `__m256d` 4-way / ARM: `float64x2_t` 2-way），只有同时开启 OPT_FLOAT 时才用 float（x86: `__m256` 8-way / ARM: `float32x4_t` 4-way）。

#### 2.5.3  GPU 五级优化

**源文件:** 所有 GPU 变体均从 `gpu_uopt_src/` 下的统一源文件编译，通过 `-D` 宏切换优化。
旧版独立源文件 (`shuanzhi/` 和 `shuanzhi_gpu_opt/`) 保留作为参考，测试脚本不再使用。

| # | 目录 | 编译宏 | RK4 | Kalman | Hungarian | EKF |
|---|------|------|:--:|:--:|:--:|:--:|
| 0 | `gpu_base/` | （无） | double, 循环内除法 | double, 循环+高斯消元 | double, 数组标记 | double, 循环 mat_mul |
| 1 | `gpu_fp32/` | `-DOPT_FP32` | float | float | float | float |
| 2 | `gpu_algo/` | `-DOPT_ALGO` | FP64+预计算常量 | FP64+展开+Cholesky+融合 | 位掩码+列规约 | FP64+展开+解析逆+__sinf |
| 3 | `gpu_mem/` | `-DOPT_MEM` | (no-op, OPT_MEM未使用) | FP64+Shared Memory | (no-op, OPT_MEM未使用) | — (ft2000未编译; bench_all未单独编译) |
| 4 | `gpu_opt/` | FP32+ALGO(+MEM) | float+预计算 | float+展开+Cholesky+SharedMem | **实际=ALGO**（脚本用cp覆盖,只用-DOPT_ALGO） | float+展开+解析逆+__sinf |

**注:**
- Kalman GPU 源码中基线为 double，并非"已是float"。所有四个算法的 GPU 源码都用 `#ifdef OPT_FP32` 切换 float。
- Hungarian GPU OPT 二进制被脚本 `cp gpu_algo/hungarian_gpu_algo gpu_opt/hungarian_gpu_opt` 覆盖，实际只有 `-DOPT_ALGO`，不含 FP32。
- RK4/Hungarian 的 `-DOPT_MEM` 在源码中未被引用，编译了 no-op 二进制。
- bench_all.sh 编译 EKF 四个变体 (base/fp32/algo/opt)，ft2000_bench.sh 仅编译 base/opt 两个。

#### 2.5.4  目录-文件-代码映射

**RK4 统一源文件 `cpu_uopt_src/rk4_uopt.c`：**

```
OPT_OPENMP    → #pragma omp parallel for 包裹外层批量循环
OPT_SIMD      → 根据 OPT_FLOAT 选择 vec_t=__m256d(4×double) 或 __m256(8×float)
                 使用 vec_load/vec_store/vec_fma/vec_neg 替代标量运算
                 尾部标量循环补齐非对齐部分
OPT_FLOAT     → typedef real_t=float, 替换 double
OPT_ALGO      → 缓存分块 BLK=256, 外层循环按 BLK 切分
```

**Kalman 统一源文件 `cpu_uopt_src/kalman_uopt.c`：**

```
OPT_OPENMP    → #pragma omp parallel for 包裹 1024 问题批量循环
OPT_SIMD      → mat_mul_7x7: NEON 4路并行前4列, 余3列标量补齐
                vfmaq_f32 积和熔合, vaddvq_f32 水平求和
OPT_FLOAT     → typedef real_t=float, R_FABS=fabsf, R_SQRT=sqrtf
OPT_ALGO      → 7×7 矩阵完全展开 (343条显式乘法)
                Cholesky 替代高斯消元 (利用S对称正定性)
                predict+correct 融合为单步 (减少临时变量)
```

**Hungarian 统一源文件 `cpu_uopt_src/hungarian_uopt.c`：**

```
OPT_OPENMP    → #pragma omp parallel for 包裹 1024 问题批量循环
OPT_SIMD      → 无效果: 算法核心是位运算, 不适用SIMD
OPT_FLOAT     → typedef real_t=float, 浮点规约在8×8矩阵上收益有限
OPT_ALGO      → 栈分配 cost[64] 替代 malloc
                位掩码 unsigned long long ms/mp 替代 char[64] 标记数组
                __builtin_ctz/__builtin_popcount 替代数组扫描
                行规约+列规约同时执行
```

**EKF 统一源文件 `cpu_uopt_src/ekf_uopt.c`：**

```
OPT_OPENMP    → #pragma omp parallel for 包裹批量循环
OPT_SIMD      → mat_mul_4x4: NEON 4路积和熔合 (完美对齐4×4维度)
                vfmaq_f32 + vmulq_f32 + vaddvq_f32
OPT_FLOAT     → typedef real_t=float, sinf/cosf/sqrtf/fabsf
OPT_ALGO      → 4×4 矩阵完全展开 (16条显式乘法)
                2×2 S矩阵解析求逆 (6次乘法+1次除法, 替代高斯消元)
```

**RK4 GPU 统一源文件 `gpu_uopt_src/rk4_gpu_uopt.cu`：**

```
OPT_FP32      → typedef real_t=float, double→float
OPT_ALGO      → 预计算 h6/h3/h2/hh2, 循环内消除除法; 使用 __device__ inline
                函数 rk4_step() 提升编译器优化空间
OPT_MEM       → (源码中未引用此宏, 编译了 no-op 二进制)
```

**Kalman GPU 统一源文件 `gpu_uopt_src/kalman_gpu_uopt.cu`：**

```
OPT_FP32      → typedef float real, double→float (源码基线为double)
OPT_ALGO      → 7×7 完全展开 (343条显式乘法), Cholesky替代高斯消元
                predict+correct融合为单步
OPT_MEM       → __shared__ float s_F[49] 缓存系统矩阵
                每block加载一次, block内128线程共享 (消除~784KB冗余全局内存读)
```

**Hungarian GPU 统一源文件 `gpu_uopt_src/hungarian_gpu_uopt.cu`：**

```
OPT_FP32      → typedef float real_t, double→float (源码基线为double)
OPT_ALGO      → 位掩码 unsigned long long ms/mp 替代数组标记
                __ffs/__ffsll/__popc CUDA硬件指令替代循环扫描
                列规约减少后续迭代次数
                所有位操作在寄存器内完成, 零全局内存访问
```

**EKF GPU 统一源文件 `gpu_uopt_src/ekf_gpu_uopt.cu`（bench_all.sh 实际使用）：**

```
OPT_FP32      → typedef float real, __sinf/__cosf/fabsf (double→float)
OPT_ALGO      → mat_mul_4x4: 4×4 手写展开 (16条显式 FMA)
                mat_mul_4x4_AtB: 循环式 (A·B^T)
                2×2 S矩阵解析求逆 (6次乘法+1次除法)
                __sinf/__cosf GPU快速数学函数
(始终启用)     → __constant__ real d_Q[16], d_R[4], d_dt (常量缓存广播读取)
```

**EKF GPU 旧版独立源文件 (保留参考，脚本不再使用):**
- `shuanzhi/ekf_demo.cu` — 基线 double, 通用循环 mat_mul
- `shuanzhi_gpu_opt/ekf_gpu_opt.cu` — 优化版, FP32 + 寄存器展开 + 解析逆

---

> **注：** 后续章节（第三章~第十五章）涵盖技术调研与库选型、算法参考（含手算示例）、算法实现详解（C代码逐行对应）、优化技术体系（CPU 11种+GPU 4算法）、测试方法、性能结果、Bug修复记录、跨平台分析等完整内容。完整文档请查看仓库中的 `shuanzhi-完整技术文档.md` 文件（约3000行）。
