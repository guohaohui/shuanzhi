/**
 * rk4_uopt.c — RK4 CPU 统一优化版 (8级优化, 编译宏切换)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef TEST_SIZE
#define TEST_SIZE 65536
#endif
#ifndef NSTEPS
#define NSTEPS 100
#endif

/* ── 精度层 ── */
#ifdef OPT_FLOAT
typedef float real_t;
#define R_EXP  expf
#define R_FABS fabsf
#else
typedef double real_t;
#define R_EXP  exp
#define R_FABS fabs
#endif

/* ── SIMD 层 ── */
#ifdef OPT_SIMD
#ifdef __AVX2__
#include <immintrin.h>
#ifdef OPT_FLOAT
#define VW 8
typedef __m256 vec_t;
#else
#define VW 4
typedef __m256d vec_t;
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#ifdef OPT_FLOAT
#define VW 4
typedef float32x4_t vec_t;
#else
#define VW 2
typedef float64x2_t vec_t;
#endif
#endif
#endif

/* ── 辅助: CPU 计时 ── */
static inline double cpu_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── ODE 定义 ── */
static inline real_t f_test(real_t t, real_t y) { (void)t; return -y; }
static inline real_t exact_sol(real_t t, real_t y0) { return y0 * R_EXP(-t); }

/* ═══════════════════════════════════════════════════════════════
 * 标量版 rk4_step / rk4sd_step
 * ═══════════════════════════════════════════════════════════════ */
#ifndef OPT_SIMD

static inline real_t rk4_step(real_t t, real_t h, real_t y) {
    real_t y0 = y;
    real_t k = f_test(t, y0);
    y = y0 + h/(real_t)6 * k;
    real_t ytmp = y0 + (real_t)0.5 * h * k;
    k = f_test(t + h/(real_t)2, ytmp);
    y += h/(real_t)3 * k;
    ytmp = y0 + (real_t)0.5 * h * k;
    k = f_test(t + h/(real_t)2, ytmp);
    y += h/(real_t)3 * k;
    ytmp = y0 + h * k;
    k = f_test(t + h, ytmp);
    y += h/(real_t)6 * k;
    return y;
}

static inline void rk4sd_step(real_t t, real_t h, real_t *y, real_t *yerr) {
    real_t yi = *y;
    real_t y1 = rk4_step(t, h, yi);
    real_t yt = rk4_step(t, h/(real_t)2, yi);
    real_t yt2 = yt;
    yt = rk4_step(t + h/(real_t)2, h/(real_t)2, yt2);
    *yerr = (real_t)8 * (real_t)0.5 * (yt - y1) / (real_t)15;
    *y = yt;
}

#endif

/* ═══════════════════════════════════════════════════════════════
 * SIMD 向量化版
 * ═══════════════════════════════════════════════════════════════ */
#ifdef OPT_SIMD
#ifdef __AVX2__
static inline vec_t vec_neg(vec_t v) {
#ifdef OPT_FLOAT
    return _mm256_sub_ps(_mm256_setzero_ps(), v);
#else
    return _mm256_sub_pd(_mm256_setzero_pd(), v);
#endif
}

static inline vec_t vec_fma(vec_t a, vec_t b, vec_t c) {
#ifdef OPT_FLOAT
    return _mm256_fmadd_ps(b, c, a);
#else
    return _mm256_fmadd_pd(b, c, a);
#endif
}

static inline vec_t vec_load(const real_t *p) {
#ifdef OPT_FLOAT
    return _mm256_loadu_ps(p);
#else
    return _mm256_loadu_pd(p);
#endif
}

static inline void vec_store(real_t *p, vec_t v) {
#ifdef OPT_FLOAT
    _mm256_storeu_ps(p, v);
#else
    _mm256_storeu_pd(p, v);
#endif
}

static inline vec_t vec_set1(real_t x) {
#ifdef OPT_FLOAT
    return _mm256_set1_ps(x);
#else
    return _mm256_set1_pd(x);
#endif
}
#endif

#ifdef __aarch64__
static inline vec_t vec_neg(vec_t v) {
#ifdef OPT_FLOAT
    return vnegq_f32(v);
#else
    return vnegq_f64(v);
#endif
}
#endif
#endif

/* ── 批量 RK4 (标量 or SIMD + 可选 OpenMP) ── */
static void rk4_batch(const real_t *y0, int n, real_t h, int steps, real_t *result) {
#ifdef OPT_ALGO
#define BLK 256
#ifdef OPT_OPENMP
#pragma omp parallel for
#endif
    for (int base = 0; base < n; base += BLK) {
        int end = base + BLK;
        if (end > n) end = n;
#ifdef OPT_SIMD
        int i;
        for (i = base; i <= end - VW; i += VW) {
            vec_t y = vec_load(&y0[i]);
            vec_t t = vec_set1(0);
            vec_t hv = vec_set1(h);
            vec_t h6 = vec_set1(h / (real_t)6);
            vec_t h3 = vec_set1(h / (real_t)3);
            vec_t h2 = vec_set1(h / (real_t)2);
            vec_t hh2 = vec_set1((real_t)0.5 * h);
            for (int s = 0; s < steps; s++) {
                vec_t y0_v = y;
                vec_t k = vec_neg(y0_v);
                y = vec_fma(y0_v, h6, k);
                vec_t ytmp = vec_fma(y0_v, hh2, k);
                k = vec_neg(ytmp);
                y = vec_fma(y, h3, k);
                ytmp = vec_fma(y0_v, hh2, k);
                k = vec_neg(ytmp);
                y = vec_fma(y, h3, k);
                ytmp = vec_fma(y0_v, hv, k);
                k = vec_neg(ytmp);
                y = vec_fma(y, h6, k);
                t = vec_fma(t, hv, vec_set1(1));
            }
            vec_store(&result[i], y);
        }
        for (; i < end; i++) {
#else
        for (int i = base; i < end; i++) {
#endif
            real_t y = y0[i];
            real_t t = 0;
            for (int s = 0; s < steps; s++) {
                y = rk4_step(t, h, y);
                t += h;
            }
            result[i] = y;
        }
    }
#undef BLK
#else
#ifdef OPT_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        real_t y = y0[i];
        real_t t = 0;
        for (int s = 0; s < steps; s++) {
            y = rk4_step(t, h, y);
            t += h;
        }
        result[i] = y;
    }
#endif
}

/* ── main ── */
int main(void) {
    int n = TEST_SIZE;
    real_t h = 0.01;
    int steps = NSTEPS;
    real_t T = steps * h;

    real_t *y0 = (real_t *)malloc(n * sizeof(real_t));
    real_t *exact = (real_t *)malloc(n * sizeof(real_t));
    real_t *result = (real_t *)malloc(n * sizeof(real_t));
    real_t *yerrs = (real_t *)malloc(n * sizeof(real_t));

    srand(42);
    for (int i = 0; i < n; i++)
        y0[i] = (real_t)0.5 + (real_t)1.5 * (real_t)rand() / (real_t)RAND_MAX;
    for (int i = 0; i < n; i++)
        exact[i] = exact_sol(T, y0[i]);

    printf("RK4 CPU n=%d h=%.2f steps=%d", n, (double)h, steps);
#ifdef OPT_OPENMP
    printf(" +OMP(%d)", omp_get_max_threads());
#endif
#ifdef OPT_SIMD
    printf(" +SIMD(VW=%d)", VW);
#endif
#ifdef OPT_FLOAT
    printf(" +Float");
#endif
#ifdef OPT_ALGO
    printf(" +Algo(Tiling)");
#endif
    printf("\n");

    double t0 = cpu_time();
    rk4_batch(y0, n, h, steps, result);
    printf("  Time: %.4f s\n", cpu_time() - t0);

    int errs = 0;
    for (int i = 0; i < n; i++) {
        double d = R_FABS(exact[i] - result[i]);
        if (d / (R_FABS(exact[i]) + 1e-15) > 1e-4) errs++;
    }
    printf("  %s\n", errs == 0 ? "ALL PASS" : "FAIL");

    free(y0); free(exact); free(result); free(yerrs);
    return 0;
}
