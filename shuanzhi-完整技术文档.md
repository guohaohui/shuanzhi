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
| 2 | `gpu_algo/` | `-DOPT_ALGO` | FP64+预计算常量 | FP64+展开+Cholesky+融合 | 位掩码+列规约 | FP64+展开+解析逆 |
| 3 | `gpu_mem/` | `-DOPT_MEM` | (no-op) | FP64+Shared Memory | (no-op) | — |
| 4 | `gpu_opt/` | FP32+ALGO(+MEM) | float+预计算 | float+展开+Cholesky+SharedMem | 实际=ALGO | float+展开+解析逆 |

#### 2.5.4  目录-文件-代码映射

**RK4 统一源文件 `cpu_uopt_src/rk4_uopt.c`：**
- OPT_OPENMP → `#pragma omp parallel for` 包裹外层批量循环
- OPT_SIMD → AVX2/NEON intrinsic 向量化 rk4_step
- OPT_FLOAT → `typedef real_t=float`
- OPT_ALGO → 缓存分块 BLK=256

**Kalman 统一源文件 `cpu_uopt_src/kalman_uopt.c`：**
- OPT_OPENMP → `#pragma omp parallel for`
- OPT_SIMD → NEON 4路并行 + AVX2 向量化 mat_mul_7x7
- OPT_ALGO → 7×7完全展开 + Cholesky替代高斯消元 + predict+correct融合

**Hungarian 统一源文件 `cpu_uopt_src/hungarian_uopt.c`：**
- OPT_ALGO → 栈分配 + 位掩码(unsigned long long) + `__builtin_ctz`/`__builtin_popcount` + 列规约
- OPT_SIMD → 无效果（算法核心是位运算）

**EKF 统一源文件 `cpu_uopt_src/ekf_uopt.c`：**
- OPT_ALGO → 4×4矩阵完全展开 + 2×2解析逆
- OPT_SIMD → NEON 4×float mat_mul_4x4

**GPU 统一源文件映射详见源文件头部注释。**

---

## 第三章  技术调研与库选型

### 3.1  调研动机

在自实现四个算法前，对业界主流数值计算库进行了全面调研。核心问题是：**是否有成熟的第三方库可以直接使用，还是需要自实现？**

选型原则: **"参考而不依赖"**——所有参考库的算法逻辑均在项目中自实现，不产生链接依赖。参考库仅作为正确性基准和设计参考。

### 3.2  RK4 ODE 求解器 — 参考 GSL

| 库 | 语言 | RK4 支持 | 误差估计 | 自适应步长 |
|------|:--:|:--:|:--:|:--:|
| **GNU GSL** ★ | C | ✅ | ✅ Step Doubling | ✅ |
| SUNDIALS ARKode | C/C++ | ✅ | ✅ | ✅ |
| Boost.Odeint | C++ | ✅ | ✅ | ✅ |
| Numerical Recipes | C/C++ | ✅ | ❌ 简化版 | ❌ |

**选 GSL 的理由：** `rk4.c` 仅 ~350 行，操作序列可直接翻译；步长折半误差估计是区别于教科书 RK4 的关键特性。

### 3.3  匈牙利算法 — 参考 Google OR-Tools

| 库 | 语言 | 算法 | 复杂度 |
|------|:--:|------|:--:|
| **Google OR-Tools** ★ | C++ | Munkres + Push-Relabel | O(n⁴) / O(n·m·log(nC)) |
| LAPJV | C | Jonker-Volgenant | O(n³) |
| dlib | C++ | 匈牙利算法 | O(n³) |

**选 OR-Tools 的理由：** 同时提供教学版 O(n⁴) 和 生产版 O(n·m·log(nC)) 两套方案；测试用例可复用。

### 3.4  卡尔曼滤波 — 参考 OpenCV

| 库 | 语言 | KF 支持 | 扩展支持 |
|------|:--:|:--:|:--:|
| **OpenCV** ★ | C++ | ✅ | ❌ 仅标准 KF |
| BFL | C++ | ✅ | ✅ EKF, PF |
| TinyEKF | C | ✅ | ✅ EKF |

**选 OpenCV 的理由：** `test_kalman.cpp` 定义精确参数/流程/容差，7维100步配置已成事实标准。

### 3.5  矩阵运算库 — 结论：自实现

对 2×2/4×4/7×7 固定小矩阵，手写展开比 BLAS/cuBLAS 更高效。通用库的调度开销对小矩阵 > 计算本身。

### 3.6  选型总结

| 算法 | 参考库 | 不直接使用的原因 | 自实现优势 |
|------|------|------|------|
| RK4 | GSL | 无 GPU, LGPL 依赖 | CPU/GPU 双模式, 零依赖 |
| Kalman | OpenCV | C++, ~50MB, 无 GPU | 固定维度7×7展开 |
| Hungarian | OR-Tools | C++, protobuf 依赖 | 位掩码+BSF位运算 |
| EKF | 无专用库 | — | 雅可比解析计算, 快速数学 |

---

## 第四章  算法参考

### 4.1  RK4 龙格-库塔法

四阶龙格-库塔法通过 4 次斜率估计的加权平均来推进一步：

```
k₁ = f(t, y₀)
k₂ = f(t + h/2, y₀ + h·k₁/2)
k₃ = f(t + h/2, y₀ + h·k₂/2)
k₄ = f(t + h, y₀ + h·k₃)

y(t+h) = y₀ + (k₁ + 2k₂ + 2k₃ + k₄) · h/6
```

局部截断误差: O(h⁵)，全局误差: O(h⁴)。

#### GSL 步长折半原理

```
路径 A: 一整步 h → y_onestep      (4 次 f 求值)
路径 B: 两个半步 h/2 → y_twostep  (8 次 f 求值)

Δ = y_twostep - y_onestep
yerr = 8.0 × 0.5 × Δ / 15 = 4/15 · Δ
```

输出采用更精确的 `y_twostep`。总计 12 次 f 求值/步（经典 RK4 的 3 倍）。

#### 手算示例

以最简单的情况演示：ODE `dy/dt = -y`, t=0, y₀=1.0, h=0.1。

```
k₁ = -1.0
k₂ = f(0.05, 0.95) = -0.95
k₃ = f(0.05, 0.9525) = -0.9525
k₄ = f(0.1, 0.90475) = -0.90475

y(0.1) = 1.0 + 0.1/6 × (-1.0 - 1.9 - 1.905 - 0.90475) ≈ 0.90484
```

解析解 `e^{-0.1} ≈ 0.904837`，误差约 **3×10⁻⁶**。

### 4.2  卡尔曼滤波

#### 线性系统模型

```
状态方程:  xₖ = F·xₖ₋₁ + wₖ    (wₖ ~ N(0, Q))
观测方程:  zₖ = H·xₖ + vₖ      (vₖ ~ N(0, R))
```

本项目参数: F=H=I₇, Q=I₇, R=σ²I₇ (σ=1.0)。

#### 预测-校正循环

```
预测:
  x̂ₖ|ₖ₋₁ = F·x̂ₖ₋₁
  Pₖ|ₖ₋₁ = F·Pₖ₋₁·Fᵀ + Q

校正:
  Kₖ = Pₖ|ₖ₋₁·Hᵀ·(H·Pₖ|ₖ₋₁·Hᵀ + R)⁻¹
  x̂ₖ = x̂ₖ|ₖ₋₁ + Kₖ·(zₖ - H·x̂ₖ|ₖ₋₁)
  Pₖ = (I - Kₖ·H)·Pₖ|ₖ₋₁
```

#### 手算示例 (简化2维)

以 dim=2, F=H=I₂, P₀=diag[1,1], z=[1.2, -0.5] 演示一个完整的预测-校正周期。卡尔曼增益使不确定性从 1.01 降到 0.502。

### 4.3  匈牙利算法 (Munkres)

#### 指派问题定义

给定 n×n 成本矩阵 C，求解一个排列 π，使得总成本 Σᵢ C[i, π(i)] 最小。时间复杂度: O(n³)。

#### Munkres 算法步骤

```
1. 行归约: 每行减去该行最小值
2. 列归约: 每列减去该列最小值
3. 试指派: 用最少的线覆盖所有 0
   a. 标记独立 0
   b. 标记被覆盖的行和列
   c. 若覆盖线数 = n → 找到最优解
   d. 否则 → 调整矩阵
4. 重复步骤 3 直到收敛
```

#### 手算示例 (3×3)

```
成本矩阵:  [4, 8, 6]     行归约后: [0, 4, 2]    列归约后: [0, 4, 0]
           [2, 5, 4]               [0, 3, 2]              [0, 3, 0]
           [7, 3, 6]               [4, 0, 3]              [4, 0, 1]

最优分配: [0, 2, 1] → 总成本 = 4 + 4 + 3 = 11
```

暴力枚举验证 (3!=6种排列) 确认 11 是全局最优。

### 4.4  扩展卡尔曼滤波 (EKF)

#### 非线性系统线性化

EKF 对非线性状态转移 f(x) 和观测 h(x) 在当前估计点进行一阶泰勒展开：

```
状态预测:  x̂ₖ|ₖ₋₁ = f(x̂ₖ₋₁)
协方差:    Pₖ|ₖ₋₁ = J_f·Pₖ₋₁·J_fᵀ + Q
校正:      Kₖ = Pₖ|ₖ₋₁·J_hᵀ·(J_h·Pₖ|ₖ₋₁·J_hᵀ + R)⁻¹
           x̂ₖ = x̂ₖ|ₖ₋₁ + Kₖ·(zₖ - h(x̂ₖ|ₖ₋₁))
           Pₖ = (I - Kₖ·J_h)·Pₖ|ₖ₋₁
```

其中 J_f = ∂f/∂x, J_h = ∂h/∂x 为雅可比矩阵。

#### 阻尼摆状态方程

状态向量 `x = [θ, ω, α, β]`：

```
θ' = θ + ω·dt
ω' = ω - α·sin(θ)·dt - β·ω·dt
α' = α    (常数参数, 待估计)
β' = β    (常数参数, 待估计)
```

观测 `z = [θ, sin(θ)]`。真值 α=9.81, β=0.30。

#### 雅可比矩阵

```
J_f = [ 1              dt             0        0     ]
      [ -α·cos(θ)·dt   1-β·dt        -sinθ·dt  -ω·dt ]
      [ 0              0              1        0     ]
      [ 0              0              0        1     ]

J_h = [ 1       0       0       0   ]
      [ cos(θ)  0       0       0   ]
```

---

## 第五章  算法实现详解

本章对照数学计算步骤，逐步讲解四个算法在本项目中的 C 语言实现。每个算法按照"**输入 → 计算步骤 → 输出 → 验证**"的流程展开。

### 5.1  RK4 / RK4SD 实现

**问题:** 求解 ODE `dy/dt = -y`，输入 n=65536 个初始点 y₀∈[0.5,2.0]，步长 h=0.01，步数 N=100。

**步骤 1 — 定义 ODE：** `f(t,y) = -y`，解析解 `y_exact(T) = y₀ · e^{-T}`

**步骤 2 — 标准 RK4 单步 (GSL 分步累加方式)：**
```c
// ① k1 step
double k = f_test(t, y0);
y = y0 + h/6.0 * k;
double ytmp = y0 + 0.5*h*k;
// ② k2 step
k = f_test(t + 0.5*h, ytmp);
y += h/3.0 * k;
ytmp = y0 + 0.5*h*k;
// ③ k3 step
k = f_test(t + 0.5*h, ytmp);
y += h/3.0 * k;
ytmp = y0 + h*k;
// ④ k4 step
k = f_test(t + h, ytmp);
y += h/6.0 * k;
```

**步骤 3 — RK4SD 步长折半：** 一整步 h (4次f) + 两个半步 h/2 (8次f) = 12次f求值。误差 `yerr = 8.0 * 0.5 * (y_twostep - y_onestep) / 15.0`。

**步骤 4 — 批量积分：** n=65536 个初始点独立完成 N=100 步积分，外层 `for(i)` 天然适合并行化。

**步骤 5 — 精度验证：** 数值解 vs 解析解，经典 RK4 tol=1e-4, RK4SD tol=1e-5。

### 5.2  卡尔曼滤波实现

**问题:** 7 维线性系统。F=H=I₇, Q=I₇, R=σ²I₇ (σ=1.0)。批量 65536 个独立问题，每问题 100 步。

**核心函数：** `kalman_predict()` → 状态预测 + 协方差预测；`kalman_update()` → 新息协方差 S + 卡尔曼增益 K (解 S·Kᵀ=H·P) + 状态/协方差更新。

**线性方程组求解：** 基线用高斯-约当消元（列主元选择）；OPT_ALGO 版用 Cholesky 分解（利用 S 对称正定性）。

**精度验证：** 无噪声场景误差=0；有噪声场景估计误差 < 原始噪声。

### 5.3  匈牙利算法实现

**问题:** 8×8 成本矩阵，批量 65536 问题。Munkres 6步法。

**关键实现：** 行归约→列归约→试指派→覆盖检测→矩阵调整。OPT_ALGO 版用位掩码 `uint64_t` + `__builtin_ctz` 替代数组扫描。

**精度验证：** OR-Tools 4×4 标准矩阵 (min=275) + 暴力枚举 n=3~6 (差异=0)。

### 5.4  EKF 实现

**问题:** 4D 阻尼摆状态估计+参数辨识。状态 `[θ,ω,α,β]`，观测 `[θ,sinθ]`。

**核心函数：** `ekf_f()` 非线性转移 → `ekf_F()` 雅可比 J_f → Predict (J_f·P·J_fᵀ+Q) → 观测 J_h → 解析 2×2 逆 → Kalman Gain → Update。

**精度验证：** 雅可比解析 vs 数值微分 (ε=1e-5)，max diff=1.36e-3 PASS。参数收敛验证：|α_est-9.81|<2.0。

---

## 第六章  优化技术体系

### 6.1  CPU 优化技术

#### 6.1.1  OpenMP 多线程并行

外层批量循环加 `#pragma omp parallel for`。x86 用 4 线程，FT2000 用 8 线程。通过 `OMP_PLACES=cores` + `OMP_PROC_BIND=close` 将线程绑定到物理核。

**多核分配机制 (i5-14600KF + 8线程为例):**
```
OMP_PLACES=cores → 10个物理核识别为10个place
OMP_PROC_BIND=close → 8线程依次绑到前8个place
线程0 → 物理核0, 线程1 → 物理核1, ..., 线程7 → 物理核7
每个物理核只跑1个线程，不共享、不争抢
```

#### 6.1.2  AVX2 SIMD 向量化

256-bit YMM 寄存器，一次处理 4 个 double 或 8 个 float：
```c
__m256d y = _mm256_cvtps_pd(y0_f4);   // float[4] → double[4]
__m256d k = _mm256_sub_pd(zero, y);   // k = -y (4路并行)
y = _mm256_fmadd_pd(h6_v, k, y0_v);   // y = y0 + h6*k (FMA融合乘加)
```

#### 6.1.3  ARM NEON SIMD

128-bit 寄存器，一次处理 4 个 float：
```c
float32x4_t y = vld1q_f32(&y0[i]);    // 加载4个float
float32x4_t k1 = vnegq_f32(y);         // k = -y
y = vfmaq_f32(y0_v, h6_v, k1);        // FMA: y = y0 + h6*k1
```

#### 6.1.4  float 替代 double

关键结论：float 单独使用几乎无收益 (~1.0×)，必须配合 SIMD 才体现价值（8-way float vs 4-way double → 2×吞吐）。

#### 6.1.5  预计算常量

```c
const float h6 = h/6.0f, h3 = h/3.0f, h2 = h/2.0f, hh2 = 0.5f*h;
```
减少 4 次除法/步 × 100 步 = 400 次除法/初始点。

#### 6.1.6  固定维度循环展开

**Kalman 7×7 mat_mul:** 完全展开为 343 次显式乘法。
**EKF 4×4 mat_mul:** 展开为 64 次显式乘法。

#### 6.1.7  缓存分块 Tiling

RK4 Full Opt 版将 65536 点分成 BLK=256 的块，整块数据在 L1 缓存中完成 100 步积分。

#### 6.1.8  Cholesky 分解替代高斯消元

Kalman Full Opt 版。Cholesky: n³/3 vs 高斯消元: 2n³/3。对 7×7: 约 50% 运算量减少。

#### 6.1.9  2×2 解析逆替代高斯消元

EKF Full Opt 版的关键优化。S 是 2×2 矩阵（观测维度固定为2）：
```c
float det = S[0]*S[3] - S[1]*S[2];
float inv_det = 1.0f / det;
S_inv[0] =  S[3] * inv_det;
S_inv[1] = -S[1] * inv_det;
S_inv[2] = -S[2] * inv_det;
S_inv[3] =  S[0] * inv_det;
```
6次乘法+1次除法完成2×2求逆。

#### 6.1.10  栈分配 + 位运算 BSF/POPCNT

Hungarian Full Opt 版。`malloc→栈分配` + `char[64]→uint64_t位掩码` + `__builtin_ctz`单周期位扫描。

#### 6.1.11  SIMD 覆盖总表

| 算法 | x86 AVX2 | ARM NEON | SIMD宽度 | 加速对象 |
|------|:--:|:--:|:--:|------|
| RK4 omp | ✅ __m256d | ✅ float32x4 | 4-double/4-float | rk4_step |
| RK4 opt | ✅ __m256 | ✅ float32x4 | 8-float/4-float | rk4_step |
| Kalman omp | ✅ __m128 | ✅ float32x4 | 4-float | mat_mul |
| Kalman opt | ✅ __m256 | ✅ float32x4 | 8-float/4-float | 7×7 mat_mul |
| EKF omp | ✅ __m128 | ✅ float32x4 | 4-float | mat_mul |
| EKF opt | ✅ __m128 | ✅ float32x4 | 4-float | 4×4 mat_mul |
| Hungarian | N/A | N/A | — | 位运算 |

### 6.2  GPU 优化技术

#### 6.2.1  RK4 优化

| 优化 | 原版 | 优化版 | 原理 |
|------|------|------|------|
| FP64→FP32 | `double y` | `float y` | 消费级GPU FP64吞吐仅为FP32的1/32~1/64 |
| 预计算常量 | 每步4次除法 | 循环外预计算h6/h3/h2/hh2 | GPU除法延迟~14周期，乘法~5周期 |

#### 6.2.2  Kalman 优化

| 优化 | 原版 | 优化版 | 原理 |
|------|------|------|------|
| 7×7完全展开 | 三重循环 | 343条显式乘法 | 消除循环开销，编译器可更好调度 |
| Cholesky替代高斯 | 高斯-约当消元 | Cholesky分解 | 利用S对称正定性，无选主元/无分支 |
| predict+correct融合 | 分离函数调用 | 单基本块完成 | 中间变量保持在寄存器，避免spill |
| Shared Memory | 每线程独立读Global | `__shared__` 每block加载一次 | ~20 cycles vs ~300 cycles |

#### 6.2.3  Hungarian 优化

| 优化 | 原版 | 优化版 | 原理 |
|------|------|------|------|
| 位掩码替代数组 | `char marks[64]` | `uint64_t` 位掩码 | 寄存器内操作，零内存访问 |
| 列规约 | 仅行规约 | 行+列规约 | 减少迭代次数 |

#### 6.2.4  EKF 优化

| 优化 | 原版 | 优化版 | 原理 |
|------|------|------|------|
| Constant Memory | kernel参数传递 | `__constant__` | 专用64KB缓存，广播读取~1 cycle |
| 4×4寄存器展开 | 循环mat_mul | 16条显式FMA | 寄存器带宽~8 TB/s |
| __sinf/__cosf | sinf/cosf | GPU SFU内建函数 | 吞吐约2× |
| 2×2解析逆 | 通用solve | `det=ac-b², inv=1/det` | 6次乘法+1次除法 |

#### 6.2.5  优化效果总览

| 算法 | 原版 GPU | 优化版 GPU | 加速比 |
|------|------:|------:|:---:|
| RK4 | 0.0006s | 0.0003s | 2.0× |
| Kalman | 0.0142s | 0.0014s | 10.1× |
| Hungarian | 0.0036s | 0.0003s | 12.0× |
| EKF | 0.0009s | 0.0001s | 9.0× |

#### 6.2.6  优化共性与差异

**共性:** 四个算法的GPU优化遵循同一原则——减少全局内存访问，增加寄存器/SM内数据重用。

**差异:** RK4瓶颈在除法单元和FP64吞吐；Kalman瓶颈在冗余内存读和分支；Hungarian瓶颈在数组扫描；EKF瓶颈在函数调用和常量传递。

### 6.3  各版本优化对比表

#### CPU 八级优化梯度

| 版本 | 编译宏 | RK4 | Kalman | Hungarian | EKF |
|------|------|------:|------:|------:|------:|
| Baseline | — | 0.042s | 6.24s | 0.13s | 1.08s |
| OMP | OPT_OPENMP | 0.011s (3.9×) | 1.73s (3.6×) | 0.039s (3.3×) | 0.26s (4.2×) |
| SIMD | OPT_SIMD | 0.009s (5.0×) | 3.24s (1.9×) | 0.14s (0.9×) | 0.65s (1.7×) |
| Float | OPT_FLOAT | 0.041s (1.0×) | 6.29s (1.0×) | 0.13s (1.0×) | 0.92s (1.2×) |
| SIMD+Float | 二者 | 0.004s (10.3×) | 3.40s (1.8×) | 0.13s (1.0×) | 0.58s (1.9×) |
| OMP+SIMD | 二者 | 0.002s (19.2×) | 1.01s (6.2×) | 0.036s (3.6×) | 0.15s (7.4×) |
| OMP+SIMD+Float | 三者 | 0.001s (32.5×) | 0.95s (6.6×) | 0.033s (3.9×) | 0.16s (6.9×) |
| Full Opt | +OPT_ALGO | 0.001s (32.5×) | 1.01s (6.2×) | 0.019s (6.8×) | 0.22s (5.0×) |

**关键结论：** OMP最稳定(3.3~4.2×)；Float单独无收益(全部≈1.0×)；SIMD对小矩阵效果弱(Kalman 7×7仅1.9×)；算法优化在Hungarian上最显著(位掩码 6.8×)。

#### GPU 五级优化梯度

| 版本 | 编译宏 | RK4 | Kalman | Hungarian | EKF |
|------|------|------:|------:|------:|------:|
| BASE | — | 0.0006s (71×) | 0.133s (47×) | 0.004s (35×) | 0.006s (172×) |
| FP32 | OPT_FP32 | 0.0006s (71×) | — | — | — |
| ALGO | OPT_ALGO | 0.0007s (60×) | 0.199s (31×) | 0.003s (52×) | — |
| MEM | OPT_MEM | 0.0006s (71×) | 0.123s (51×) | — | — |
| OPT | 全开 | 0.0002s (212×) | 0.011s (558×) | 0.003s (50×) | 0.0005s (2166×) |

**关键结论：** Kalman GPU OPT加速最大(558×)：double→float + Shared Memory + Cholesky三条叠加；GPU ALGO单独可能变慢(Kalman ALGO 0.199s比BASE 0.133s慢)；EKF GPU OPT 2166×：4×4寄存器展开+解析逆+__sinf+Constant Memory四重叠加。

---

## 第七章  逐算法优化详解

> **本章内容已合并到第六章。** 详见第六章 6.1 节（CPU 11种优化）和 6.2 节（GPU 4算法优化）。

---

## 第八章  测试方法与参数

### 8.1  输入参数总表

| 参数 | RK4 | RK4SD | Kalman | Hungarian | EKF |
|------|:---:|:---:|:---:|:---:|:---:|
| 规模 | n=65536 | n=65536 | dim=7, batch=65536 | n=8, batch=65536 | dim=4, batch=65536 |
| 步数 | steps=100 | steps=100 | steps=100 | — | steps=100 |
| 步长 | h=0.01, T=1.0 | h=0.01, T=1.0 | — | — | dt=0.01 |
| 模型 | dy/dt=-y | dy/dt=-y | F=H=I, Q=I | 成本∈[1,100] | 阻尼摆 |
| 噪声 | 无 | 无 | σ=1.0 | 无 | Q/R配置 |
| 随机种子 | 42 | 42 | 42 | 42 | 42/123 |

### 8.2  测试流程

1. 编译所有版本 (Baseline / OMP+SIMD / Full Opt / GPU各版)
2. 设置 OMP_NUM_THREADS
3. 依次运行每个版本，记录耗时+验证正确性
4. 计算加速比: Speedup = T_baseline / T_optimized

### 8.3  计时方法

CPU: `clock_gettime(CLOCK_MONOTONIC)` / `omp_get_wtime()`
GPU: `cudaEventRecord` + `cudaEventElapsedTime`

### 8.4  正确性验证方法

| 算法 | 验证方法 | 容差 |
|------|---------|:--:|
| RK4 | 数值解 vs 解析解 y=y₀·e^{-T} | 1e-4 |
| RK4SD | 同RK4 + 误差估计统计 | 1e-5 |
| Kalman | 无噪声场景误差=0; 有噪声场景误差<原始噪声 | 1e-4 / 动态 |
| Hungarian | OR-Tools 4×4(成本=275) + 暴力枚举n=3~6 | 精确匹配 |
| EKF | 雅可比解析vs数值微分; 参数收敛到真值 | 1.5e-3 / 2.0 |

### 8.5  编译命令参考

#### CPU 编译模板
```bash
gcc -O2 {宏定义} cpu_uopt_src/{算法}_uopt.c -lm \
    -DTEST_SIZE=65536 -DBATCH_SIZE=65536 -DMATRIX_SIZE=8 \
    -o {输出目录}/{算法}_{版本后缀}
```

#### GPU 编译模板
```bash
nvcc -arch=sm_XX -O2 {宏定义} gpu_uopt_src/{算法}_gpu_uopt.cu \
     -DTEST_SIZE=65536 -DBATCH_SIZE=65536 -DMATRIX_SIZE=8 \
     -o {输出目录}/{算法}_gpu_{版本后缀}
```

---

## 第九章  性能测试结果

### 9.1  测试环境

i5-14600KF + RTX 4060 Ti (x86_64, AVX2+FMA, sm_89), OMP_NUM_THREADS=4, batch=65536, -O2

### 9.2  CPU 八级优化梯度

（数据同 6.3 节 CPU 表）

### 9.3  GPU 优化梯度

（数据同 6.3 节 GPU 表）

### 9.4  完整梯度分析

#### 9.4.1  RK4 (n=65536) — kernel launch开销天花板

GPU加速比在RK4上受限(仅104×)，因为65536线程×100步的计算量(~26M flop)在RTX 4060 Ti上只需~0.3ms，kernel launch开销(~0.1ms)和PCIe传输(~0.05ms)已占显著比例。

#### 9.4.2  Kalman (batch=65536) — FP32是最大赢家

GPU BASE 0.133s → GPU FP32 0.019s (**7×**)。P400上FP64→FP32加速比可达32×。

#### 9.4.3  Hungarian (batch=65536) — 位运算不靠浮点

Hungarian GPU加速比最小(35×→50×)，因为Munkres算法核心是分支+位运算，浮点吞吐量不是瓶颈。位掩码优化(ALGO)比精度降级(FP32)更有效。

#### 9.4.4  EKF (batch=65536) — 小矩阵+ConstMem的完美风暴

EKF GPU OPT 2166×：4×4矩阵完美适配寄存器(16 floats→16 registers) + 2×2解析逆(O(1)) + Constant Memory广播 + __sinf/__cosf快速数学。

### 9.5  关键结论

1. CPU OMP 最稳定: 3.3~4.2×
2. float 独立使用无收益，必须配合SIMD
3. GPU FP64→FP32 收益最大 (消费级GPU人为限制FP64)
4. 算法优化(Hungarian位掩码/EKF解析逆)是"免费"的性能杠杆
5. 小矩阵(<100 flop/问题)不适合GPU

---

## 第十章  发现的问题与修复

### 10.1  AVX2 RK4SD 寄存器溢出 (Bug #1)

**现象:** RK4SD CPU OMP+SIMD版本(AVX2)耗时0.0838s，与Baseline(0.0852s)几乎相同——4线程+SIMD零收益。禁用AVX2后标量+OpenMP: 0.0218s(3.9×)。

**根因:** 步长折半需要在一个时间步内完成3次完整RK4(一整步+两半步)，AVX2代码展开后需要~20个YMM寄存器，但AVX2只有16个。多余寄存器spill到栈，产生大量store/load流量。

**修复:** RK4SD禁用AVX2路径，改用标量rk4sd_step+OpenMP多线程。

**教训:** SIMD有适用条件——寄存器不够时反而有害。GPU有255个寄存器/线程，无此问题。

### 10.2  EKF GPU原版随机种子不匹配 (Bug #2)

**现象:** GPU版本与CPU版本结果不同(srand调用在GPU kernel之前)。

**根因:** `srand()`未在GPU数据生成前重置，导致GPU和CPU处理不同的输入数据。

**修复:** 用`memcpy`从CPU版本复制相同的初始状态到GPU缓冲区。

### 10.3  Hungarian GPU原版小矩阵性能倒挂 (Bug #3)

**现象:** GPU比CPU慢(0.53×)，1024个问题GPU 0.0036s vs CPU 0.0019s。

**根因:** 8×8矩阵计算量太小(~1500 flop/问题)，kernel launch开销(~5μs)占主导。Munkres的while/if分支导致warp divergence。

**修复:** 位掩码+__ffs替代数组扫描+warp级并行。

**教训:** 小问题不适合GPU。当单问题计算量<10000 flop时，kernel launch开销可能超过计算本身。

### 10.4  问题与修复总表

| # | 问题 | 算法 | 版本 | 根因 | 修复 | 效果 |
|:--:|------|:--:|------|------|------|:--:|
| 1 | AVX2寄存器溢出 | RK4SD | CPU OMP+SIMD | 3×展开需~20 YMM | 标量+OpenMP | 1.0×→3.9× |
| 2 | 随机种子不匹配 | EKF | GPU原版 | srand未重置 | memcpy同组初始状态 | FAIL→PASS |
| 3 | 小矩阵kernel开销 | Hungarian | GPU原版 | launch>计算 | warp并行+位运算 | 0.5×→9.4× |
| 4 | RK4 GPU RK4SD ytmp基准 | RK4 | GPU BASE/FP32/MEM | 第二半步从已修改值计算 | 保存yt2起始状态 | 6.65e-04→3.85e-12 |
| 5 | Kalman NEON double缺列 | Kalman | CPU SIMD double | NEON VW=2跳过列2,3 | 增加第二段SIMD加载 | NaN→PASS |
| 6 | EKF AtB实现错误 | EKF | CPU Baseline/FullOpt | Aᵀ·B替代A·Bᵀ | 修正索引访问模式 | NaN发散→收敛 |
| 7 | EKF CPU Full Opt性能倒退 | EKF | CPU Full Opt | #if OPT_ALGO优先级bug | 修复宏条件顺序 | 0.1972→0.1571s |
| 8 | Hungarian GPU测试数据错误 | Hungarian | 全部GPU版本 | OR-Tools矩阵行序反转 | 修正测试数据 | 成本265→275 |

> 注: Bug #1-#3来自x86_64开发环境，Bug #4-#8来自FT2000跨平台测试。

---

## 第十一章  跨平台 — FT2000 + P400

### 11.1  平台差异总览

| | x86_64 | ARM64 FT2000 |
|------|------|------|
| CPU架构 | x86_64, Raptor Lake, 3.5GHz, 14核 | aarch64, ARMv8-A, 64核 |
| 单核性能 | 基准 (1.0×) | ~1/3 i5 |
| SIMD | AVX2+FMA (256-bit) | ARM NEON (128-bit) |
| GPU | RTX 4060 Ti (sm_89, 4352核, 8GB) | Quadro P400 (sm_61, 256核, 2GB) |
| GPU算力 | 基准 (1.0×) | ~1/15 RTX 4060 Ti |
| CUDA版本 | 12.6 | 11.x (P400不支持12.x) |

### 11.2  编译与运行

```bash
# 基础环境
sudo apt install -y gcc g++ make
# NVIDIA驱动 (P400需>=470.x)
# CUDA Toolkit 11.x (勿用12.x! sm_61在12.x已废弃)

# 一键测试
bash ft2000_bench.sh  # OMP_NUM_THREADS=8
```

### 11.3  ARM NEON 优化要点

NEON是128-bit，一次处理4×float或2×double。FMA指令`vfmaq_f32(acc,a,b)`执行`acc+a*b`。ARMv8有专用取负指令`vnegq_f32`和水平求和`vaddvq_f32`。

7×7矩阵在NEON上需要分段处理：VW=4覆盖0-3列，余量3列标量补齐；VW=2需两段SIMD(0-1,2-3)+余量标量。

---

## 第十二章  关键经验与结论

### 12.1  CPU 优化收益排序

| 排名 | 优化 | 典型收益 | 适用算法 | 投入 |
|:--:|------|:--:|------|:--:|
| 🥇 | OpenMP | 3~5× | 全部 | 极低(一行pragma) |
| 🥈 | 算法优化(ALGO) | 1.5~3× | 各算法特异 | 中等 |
| 🥉 | SIMD向量化 | 1.5~2.5× | 计算规则密集 | 高 |
| 4 | float+SIMD | 2.0× | 精度允许场景 | 低 |
| 5 | 缓存分块 | 1.5~2× | 大数据量 | 中 |

### 12.2  GPU 优化收益排序

| 排名 | 优化 | P400典型收益 | 适用算法 |
|:--:|------|:--:|------|
| 🥇 | FP64→FP32 | 6.5~37× | 全部(FP64瓶颈者优先) |
| 🥈 | 寄存器展开 | 2.0~3.0× | Kalman, EKF |
| 🥉 | Shared/Const Memory | 2.5~4.0× | Kalman, EKF |
| 4 | 位掩码+硬件指令 | 2.1× | Hungarian |
| 5 | 解析逆+快速数学 | 1.5~2.0× | EKF |

### 12.3  GPU vs CPU 选择决策树

```
问题规模 > 10000? ─ 否 → CPU Full Opt
  └─ 是 → 计算是否规则(无分支)?
           ├─ 是 → GPU + Shared/Const Memory
           └─ 否 → CPU Full Opt 更划算
```

### 12.4  各优化技术的投入产出比

| 优化技术 | 实现难度 | 典型收益 | 适用算法 | 推荐度 |
|------|:--:|:--:|------|:--:|
| OpenMP parallel for | ★☆☆☆☆ | 2~4× | 所有 | ⭐⭐⭐⭐⭐ |
| float替代double | ★☆☆☆☆ | 1.5~2× | RK4, Kalman | ⭐⭐⭐⭐⭐ |
| 预计算常量 | ★☆☆☆☆ | 1.1~1.3× | RK4 | ⭐⭐⭐⭐ |
| 循环展开(固定维度) | ★★☆☆☆ | 1.5~2× | Kalman, EKF | ⭐⭐⭐⭐⭐ |
| AVX2/NEON SIMD | ★★★☆☆ | 1.5~2.5× | RK4, Kalman, EKF | ⭐⭐⭐⭐ |
| 缓存分块Tiling | ★★★☆☆ | 1.5~2× | RK4 | ⭐⭐⭐ |
| Cholesky替代高斯 | ★★☆☆☆ | 1.1~1.2× | Kalman | ⭐⭐⭐⭐ |
| 2×2解析逆 | ★★☆☆☆ | 1.5× | EKF | ⭐⭐⭐⭐⭐ |
| 栈分配+位运算 | ★★★☆☆ | 2~3× | Hungarian | ⭐⭐⭐⭐ |
| GPU Shared Memory | ★★★★☆ | 3~4× | Kalman | ⭐⭐⭐⭐ |
| GPU Constant Memory | ★★☆☆☆ | 2~3× | EKF | ⭐⭐⭐⭐⭐ |
| GPU快速数学 | ★☆☆☆☆ | 1.3× | EKF | ⭐⭐⭐⭐⭐ |

### 12.5  SIMD不是银弹

- **何时用SIMD:** 内循环体<16条向量指令，寄存器压力<16
- **何时不用:** 循环体需要展开3×以上→先评估寄存器压力；若>16需spill→标量+多线程可能更优
- **GPU不同:** GPU有255个寄存器/线程，寄存器压力几乎不是问题

### 12.6  最终推荐方案

| 场景 | 推荐版本 | 说明 |
|------|------|------|
| 开发/调试 | CPU Baseline | 最简单,串行,易调试 |
| 生产部署(x86) | CPU Full Opt (4T) | 5~15×加速,零依赖 |
| 生产部署(GPU) | GPU优化版 | 10~200×加速 |
| 高精度需求 | RK4SD Full Opt | 精度1e-6~1e-8,误差估计 |
| 嵌入式ARM | CPU Full Opt (NEON) | NEON自动启用,适配FT2000 |
| 实时系统(<1μs) | EKF GPU优化版 | 0.06μs/problem |

---

## 第十三章  FT2000 测试故障修复记录

> **详细内容已合并到第十章。** FT2000平台Bug #4-#8的分析、修复和验证见第十章10.4-10.9节。

---

## 第十四章  FT2000 测试结果分析 (2026-06-02)

### 14.1  测试环境

FT2000/64 (ARMv8-A, NEON) + Quadro P400 (sm_61, 256核, 2GB) + Kylin V10。GCC 7.x, NVCC 11.x。

### 14.2  关键硬件特性

| 特性 | 数值 | 影响 |
|------|:--:|------|
| CPU核数 | 64 ARM核 | OMP可扫描8~64最优线程数 |
| NEON宽度 | 128-bit | 4×float或2×double |
| P400 FP32 | 0.64 TFLOPS | GPU加速受限 |
| P400 FP64 | 0.02 TFLOPS | FP64→FP32收益极大(32×) |
| P400 显存 | 2GB | batch不能超过~1M |

### 14.3  完整测试结果

#### RK4 (n=65536)

| 版本 | 耗时 | 加速比 | 验证 |
|------|------|:--:|------|
| CPU Baseline | 5.1178s | 1.0× | PASS |
| CPU OMP(8T) | 1.9659s | 2.6× | PASS |
| CPU Full Opt | 0.5564s | 9.2× | PASS |
| GPU BASE | 0.0078s | 656× | PASS |
| GPU OPT | 0.0077s | 664× | PASS |

#### Kalman (dim=7, batch=65536)

| 版本 | 耗时 | 加速比 | 验证 |
|------|------|:--:|------|
| CPU Baseline | 22.7345s | 1.0× | PASS |
| CPU OMP(8T) | 4.5275s | 5.0× | PASS |
| CPU Full Opt | 2.3059s | 9.9× | PASS |
| GPU BASE | 0.3140s | 72× | PASS |
| GPU OPT | 0.0755s | 301× | PASS |

#### Hungarian (n=8, batch=65536)

| 版本 | 耗时 | 加速比 | 验证 |
|------|------|:--:|------|
| CPU Baseline | 0.3944s | 1.0× | PASS |
| CPU OMP(8T) | 0.1134s | 3.5× | PASS |
| CPU Full Opt | 0.0807s | 4.9× | PASS |
| GPU BASE | 0.0435s | 9.1× | PASS |
| GPU OPT | 0.0420s | 9.4× | PASS |

#### EKF (dim=4, batch=65536)

| 版本 | 耗时 | 加速比 | 验证 |
|------|------|:--:|------|
| CPU Baseline | 3.4459s | 1.0× | PASS |
| CPU OMP(8T) | 0.6848s | 5.0× | PASS |
| CPU Full Opt | 0.1571s | 21.9× | PASS |
| GPU BASE | 0.0208s | 166× | PASS |
| GPU OPT | 0.0052s | 664× | PASS |

### 14.4  加速比合理性总结

P400(256核,FP32 0.64TFLOPS) vs FT2000(64核ARM)的单维度贡献：并行度(~50-100×) × 精度降级(10-32×) × 算法优化(2-3×) = 理论上限~1000-6000×。实测664×(EKF GPU OPT)在合理范围内。

---

## 第十五章  x86_64 测试结果 (2026-06-02, 全部修复后)

### 15.1  测试环境

i5-14600KF (6P+8E, 3.5GHz) + RTX 4060 Ti (sm_89) + Ubuntu 20.04 WSL2。GCC 9.4, NVCC 12.6。OMP_NUM_THREADS=4。

### 15.2  关键平台差异

| 特性 | x86_64 | FT2000 | 影响 |
|------|:--:|:--:|------|
| 单核IPC | 1.0× | ~0.3× | CPU Baseline差异3× |
| SIMD | AVX2 256-bit | NEON 128-bit | float吞吐差2× |
| GPU FP32 | 22 TFLOPS | 0.64 TFLOPS | GPU加速比差34× |
| GPU架构 | Ada Lovelace | Pascal | sm_89 vs sm_61 |

### 15.3  完整测试结果

（详见 `测试结果汇总_20260601.md` — 全部50版本, 152 PASS, 0 FAIL）

**RK4最大加速:** GPU OPT 0.0004s (CPU Baseline 0.0417s → 104.2×)
**Kalman最大加速:** GPU OPT 0.0080s (CPU Baseline 6.2104s → 777×)
**Hungarian最大加速:** GPU OPT 0.0090s (CPU Baseline 0.1290s → 14×...GPU OPT最终45× vs CPU)
**EKF最大加速:** GPU OPT 0.0006s (CPU Baseline 1.0800s → 267× via GPU, or CPU Full Opt 0.1571s → 6.9×)

### 15.4  CPU 最佳版本加速比汇总

| 算法 | CPU Baseline | CPU Full Opt | 加速比 |
|------|------|------|:--:|
| RK4 | 0.0417s | 0.0013s | 32.1× |
| Kalman | 6.2104s | 0.9506s | 6.5× |
| Hungarian | 0.1290s | 0.0191s | 6.8× |
| EKF | 1.0800s | 0.1571s | 6.9× |

### 15.5  GPU 最佳版本加速比汇总

| 算法 | CPU Baseline | GPU OPT | 加速比 |
|------|------|------|:--:|
| RK4 | 0.0417s | 0.0004s | 104.2× |
| Kalman | 6.2104s | 0.0080s | 777× |
| Hungarian | 0.1290s | 0.0090s | 45× (GPU) |
| EKF | 1.0800s | 0.0006s | 267× (GPU) |

---

**文档结束。** 版本 2.3, 2026-06-02。
