# 项目环境约束

## 生产环境

此项目代码必须能在以下生产环境编译和运行：

- **CPU**: Phytium FT2000/64 (ARMv8-A, 64 cores, NEON SIMD)
- **GPU**: NVIDIA Quadro P400 (256 CUDA Cores, Pascal sm_61, 2GB)
- **OS**: Kylin V10 (aarch64)
- **编译器**: GCC (ARM64) + NVCC (sm_61)

## 开发环境

- **CPU**: Intel i5-14600KF (x86_64, AVX2+FMA)
- **GPU**: NVIDIA GeForce RTX 4060 Ti (sm_89)
- **OS**: Ubuntu (x86_64)

## 规则

1. **平台兼容**: 所有 C/CUDA 代码必须同时兼容 ARM64 (NEON) 和 x86_64 (AVX2+FMA)，通过 `#ifdef __aarch64__` / `#ifdef __x86_64__` 条件编译
2. **GPU 兼容**: CUDA 代码必须兼容 sm_61 (P400)，不能使用 sm_61 不支持的特性（如 __shfl_sync 某些变体、超过 2GB 显存）
3. **修改代码后**: 同步更新 `shuanzhi-完整技术文档.md`
4. **测试验证**: 在开发环境测试通过后，使用 `ft2000_bench.sh` 在生产环境验证
