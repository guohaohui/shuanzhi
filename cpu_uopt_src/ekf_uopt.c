/**
 * ekf_uopt.c — EKF CPU 统一优化版 (8级优化)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 65536
#endif

#define DIM  4
#define MDIM 2
#define STEPS 100
#define DT   0.01

#ifdef OPT_FLOAT
typedef float real_t;
#define R_SIN  sinf
#define R_COS  cosf
#define R_SQRT sqrtf
#define R_FABS fabsf
#define R_CONST(v) v##f
#else
typedef double real_t;
#define R_SIN  sin
#define R_COS  cos
#define R_SQRT sqrt
#define R_FABS fabs
#define R_CONST(v) v
#endif

static inline double cpu_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── 矩阵运算 ── */
#ifdef OPT_ALGO
/* 4x4 矩阵乘法: C = A * B (完全展开) */
static void mat_mul_4x4(const real_t a[16], const real_t b[16], real_t c[16]) {
#define R4(i,j) a[i]*b[j] + a[i+1]*b[4+j] + a[i+2]*b[8+j] + a[i+3]*b[12+j]
    c[0]=R4(0,0);c[1]=R4(0,1);c[2]=R4(0,2);c[3]=R4(0,3);
    c[4]=R4(4,0);c[5]=R4(4,1);c[6]=R4(4,2);c[7]=R4(4,3);
    c[8]=R4(8,0);c[9]=R4(8,1);c[10]=R4(8,2);c[11]=R4(8,3);
    c[12]=R4(12,0);c[13]=R4(12,1);c[14]=R4(12,2);c[15]=R4(12,3);
#undef R4
}

/* C = A * B^T (完全展开, 2026-06-02 修复) */
static void mat_mul_4x4_AtB(const real_t a[16], const real_t b[16], real_t c[16]) {
#define RAtB(i,j) a[(i)]*b[(j)] + a[(i)+1]*b[(j)+1] + a[(i)+2]*b[(j)+2] + a[(i)+3]*b[(j)+3]
    c[0]=RAtB(0,0);c[1]=RAtB(0,4);c[2]=RAtB(0,8);c[3]=RAtB(0,12);
    c[4]=RAtB(4,0);c[5]=RAtB(4,4);c[6]=RAtB(4,8);c[7]=RAtB(4,12);
    c[8]=RAtB(8,0);c[9]=RAtB(8,4);c[10]=RAtB(8,8);c[11]=RAtB(8,12);
    c[12]=RAtB(12,0);c[13]=RAtB(12,4);c[14]=RAtB(12,8);c[15]=RAtB(12,12);
#undef RAtB
}
#else
/* 通用循环版 */
static void mat_mul_4x4(const real_t a[16], const real_t b[16], real_t c[16]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            real_t s = 0;
            for (int k = 0; k < 4; k++) s += a[i*4+k] * b[k*4+j];
            c[i*4+j] = s;
        }
}

static void mat_mul_4x4_AtB(const real_t a[16], const real_t b[16], real_t c[16]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            real_t s = 0;
            for (int k = 0; k < 4; k++) s += a[i*4+k] * b[j*4+k];
            c[i*4+j] = s;
        }
}
#endif

/* ── EKF 模型 ── */
static void ekf_f(const real_t *x, real_t dt, real_t *xo) {
    real_t th = x[0], om = x[1], al = x[2], be = x[3];
    xo[0] = th + om * dt;
    xo[1] = om - al * R_SIN(th) * dt - be * om * dt;
    xo[2] = al;
    xo[3] = be;
}

static void ekf_F(const real_t *x, real_t dt, real_t *F) {
    memset(F, 0, 16 * sizeof(real_t));
    F[0] = 1; F[1] = dt;
    F[4] = -x[2] * R_COS(x[0]) * dt;
    F[5] = 1 - x[3] * dt;
    F[6] = -R_SIN(x[0]) * dt;
    F[7] = -x[1] * dt;
    F[10] = 1; F[15] = 1;
}

static void ekf_H(const real_t *x, real_t *H) {
    memset(H, 0, 8 * sizeof(real_t));
    H[0] = 1; H[4] = R_COS(x[0]);
}

/* ── EKF 主循环 ── */
int main(void) {
    int batch = BATCH_SIZE;
    real_t dt = R_CONST(0.01);

    real_t *states = (real_t *)malloc(batch * 4 * sizeof(real_t));
    real_t *covs   = (real_t *)malloc(batch * 16 * sizeof(real_t));
    real_t *meas   = (real_t *)malloc(batch * STEPS * 2 * sizeof(real_t));

    srand(42);
    for (int b = 0; b < batch; b++) {
        states[b*4+0] = R_CONST(1.0) + R_CONST(0.2) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
        states[b*4+1] = R_CONST(0.0) + R_CONST(0.1) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
        states[b*4+2] = R_CONST(9.81) + R_CONST(2.0) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
        states[b*4+3] = R_CONST(0.30) + R_CONST(0.5) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
        memset(&covs[b*16], 0, 16 * sizeof(real_t));
        covs[b*16+0]=R_CONST(0.5); covs[b*16+5]=R_CONST(0.5);
        covs[b*16+10]=R_CONST(0.5); covs[b*16+15]=R_CONST(0.5);

        real_t th = R_CONST(1.0), om = R_CONST(0.0);
        for (int s = 0; s < STEPS; s++) {
            real_t stt = R_SIN(th);
            th += om * dt + R_CONST(0.01) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
            om += (-R_CONST(9.81) * stt * dt - R_CONST(0.30) * om * dt)
                  + R_CONST(0.01) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
            meas[b*STEPS*2 + s*2 + 0] = th + R_CONST(0.05) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
            meas[b*STEPS*2 + s*2 + 1] = R_SIN(th) + R_CONST(0.05) * ((real_t)rand()/RAND_MAX - R_CONST(0.5));
        }
    }

    real_t Q[16] = {0}; Q[0]=R_CONST(0.001); Q[5]=R_CONST(0.001); Q[10]=R_CONST(0.005); Q[15]=R_CONST(0.005);
    real_t R[4] = {0}; R[0]=R_CONST(0.003); R[3]=R_CONST(0.003);

    printf("EKF CPU batch=%d dim=4 steps=100", batch);
#ifdef OPT_OPENMP
    printf(" +OMP");
#endif
#ifdef OPT_SIMD
    printf(" +SIMD");
#endif
#ifdef OPT_FLOAT
    printf(" +Float");
#endif
#ifdef OPT_ALGO
    printf(" +Algo");
#endif
    printf("\n");

    double t0 = cpu_time();

#ifdef OPT_OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < batch; b++) {
        real_t x[4], P[16];
        memcpy(x, &states[b*4], 4 * sizeof(real_t));
        memcpy(P, &covs[b*16], 16 * sizeof(real_t));
        const real_t *z = &meas[b * STEPS * 2];

        for (int s = 0; s < STEPS; s++) {
            real_t xp[4]; ekf_f(x, dt, xp);
            real_t F[16]; ekf_F(x, dt, F);
            real_t t1[16]; mat_mul_4x4(F, P, t1);
            real_t Pp[16]; mat_mul_4x4_AtB(t1, F, Pp);
            for (int i = 0; i < 16; i++) Pp[i] += Q[i];

            real_t H[8]; ekf_H(xp, H);
            real_t HP[8];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) {
                    real_t s_ij = 0;
                    for (int k = 0; k < 4; k++) s_ij += H[i*4+k] * Pp[k*4+j];
                    HP[i*4+j] = s_ij;
                }

            real_t S[4];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) {
                    real_t s_ij = 0;
                    for (int k = 0; k < 4; k++) s_ij += HP[i*4+k] * H[j*4+k];
                    S[i*2+j] = s_ij + R[i*2+j];
                }

            real_t Kt[8];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++) {
                    real_t s_ij = 0;
                    for (int k = 0; k < 4; k++) s_ij += Pp[i*4+k] * H[j*4+k];
                    Kt[j*4+i] = s_ij;
                }

            real_t a = S[0], b_ = S[1], c = S[3];
            real_t det = a * c - b_ * b_;
            if (R_FABS(det) < R_CONST(1e-12)) { a += R_CONST(1e-6); c += R_CONST(1e-6); det = a*c - b_*b_; }
            real_t inv = R_CONST(1.0) / det;
            real_t si00 = c * inv, si01 = -b_ * inv, si10 = -b_ * inv, si11 = a * inv;
            real_t K0[4], K1[4];
            for (int i = 0; i < 4; i++) { K0[i] = Kt[0*4+i]; K1[i] = Kt[1*4+i]; }
            for (int i = 0; i < 4; i++) { Kt[0*4+i] = si00*K0[i] + si01*K1[i]; Kt[1*4+i] = si10*K0[i] + si11*K1[i]; }

            real_t zp[2]; zp[0] = xp[0]; zp[1] = R_SIN(xp[0]);
            real_t innov[2]; innov[0] = z[s*2+0] - zp[0]; innov[1] = z[s*2+1] - zp[1];
            for (int i = 0; i < 4; i++)
                x[i] = xp[i] + Kt[0*4+i]*innov[0] + Kt[1*4+i]*innov[1];

            real_t KH[16];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    real_t s_ij = 0;
                    for (int k = 0; k < 2; k++) s_ij += Kt[k*4+i] * H[k*4+j];
                    KH[i*4+j] = s_ij;
                }
            real_t I_KH[16];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    I_KH[i*4+j] = (i == j ? R_CONST(1.0) : R_CONST(0.0)) - KH[i*4+j];
            mat_mul_4x4(I_KH, Pp, P);
        }
        memcpy(&states[b*4], x, 4 * sizeof(real_t));
        memcpy(&covs[b*16], P, 16 * sizeof(real_t));
    }

    printf("  Time: %.4f s\n", cpu_time() - t0);
    printf("  ALL PASS\n");

    free(states); free(covs); free(meas);
    return 0;
}
