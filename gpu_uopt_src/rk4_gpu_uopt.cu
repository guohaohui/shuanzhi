/* rk4_gpu_uopt.cu — RK4 GPU unified optimization (see GPU优化详解.md for details) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef TEST_SIZE
#define TEST_SIZE 65536
#endif
#ifndef NSTEPS
#define NSTEPS 100
#endif

#ifdef OPT_FP32
typedef float real_t;
#define R_EXP  expf
#define R_FABS fabsf
#else
typedef double real_t;
#define R_EXP  exp
#define R_FABS fabs
#endif

static inline float gpu_elapsed(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms / 1000.0f;
}

__device__ __host__ static inline real_t f_test(real_t t, real_t y) { (void)t; return -y; }
__device__ __host__ static inline real_t exact_sol(real_t t, real_t y0) { return y0 * R_EXP(-t); }

#ifdef OPT_ALGO
__device__ static inline real_t rk4_step(real_t t, real_t h, real_t y, real_t h6, real_t h3, real_t h2, real_t hh2) {
    (void)h; real_t y0 = y;
    real_t k = f_test(t, y0);
    y = y0 + h6 * k; real_t ytmp = y0 + hh2 * k;
    k = f_test(t + h2, ytmp); y += h3 * k; ytmp = y0 + hh2 * k;
    k = f_test(t + h2, ytmp); y += h3 * k; ytmp = y0 + h * k;
    k = f_test(t + h, ytmp); y += h6 * k;
    return y;
}

__global__ void rk4_kernel(const real_t *y0, real_t *result, real_t h, int steps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real_t y = y0[i], t = 0;
    real_t h6 = h/6, h3 = h/3, h2 = h/2, hh2 = 0.5*h;
    for (int s = 0; s < steps; s++) { y = rk4_step(t, h, y, h6, h3, h2, hh2); t += h; }
    result[i] = y;
}

__global__ void rk4sd_kernel(const real_t *y0, real_t *result, real_t *yerrs, real_t h, int steps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real_t y = y0[i], t = 0, ye = 0;
    real_t h6 = h/6, h3 = h/3, h2 = h/2, hh2 = 0.5*h;
    real_t hh = h * 0.5, sh6 = hh/6, sh3 = hh/3, sh2 = hh*0.5, shh2 = 0.5*hh;
    for (int s = 0; s < steps; s++) {
        real_t yi = y;
        real_t y1 = rk4_step(t, h, yi, h6, h3, h2, hh2);
        real_t yt = rk4_step(t, hh, yi, sh6, sh3, sh2, shh2);
        yt = rk4_step(t + hh, hh, yt, sh6, sh3, sh2, shh2);
        ye = 8 * 0.5 * (yt - y1) / 15;
        y = yt; t += h;
    }
    result[i] = y; yerrs[i] = ye;
}
#else
__global__ void rk4_kernel(const real_t *y0, real_t *result, real_t h, int steps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real_t y = y0[i], t = 0;
    for (int s = 0; s < steps; s++) {
        real_t y0_v = y;
        real_t k = f_test(t, y0_v);
        y = y0_v + h/6 * k; real_t ytmp = y0_v + 0.5*h * k;
        k = f_test(t + h/2, ytmp); y += h/3 * k; ytmp = y0_v + 0.5*h * k;
        k = f_test(t + h/2, ytmp); y += h/3 * k; ytmp = y0_v + h * k;
        k = f_test(t + h, ytmp); y += h/6 * k;
        t += h;
    }
    result[i] = y;
}

__global__ void rk4sd_kernel(const real_t *y0, real_t *result, real_t *yerrs, real_t h, int steps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real_t y = y0[i], t = 0, ye = 0;
    for (int s = 0; s < steps; s++) {
        real_t yi = y, y0_v, ytmp;
        /* full step */
        real_t k = f_test(t, yi);
        real_t y1 = yi + h/6*k; real_t yt1 = yi + 0.5*h*k;
        k = f_test(t + h/2, yt1); y1 += h/3*k; yt1 = yi + 0.5*h*k;
        k = f_test(t + h/2, yt1); y1 += h/3*k; yt1 = yi + h*k;
        k = f_test(t + h, yt1); y1 += h/6*k;
        /* half step 1 */
        real_t hh = h * 0.5; real_t yt = yi;
        k = f_test(t, yt); yt = yi + hh/6*k; yt1 = yi + 0.5*hh*k;
        k = f_test(t + hh/2, yt1); yt += hh/3*k; yt1 = yi + 0.5*hh*k;
        k = f_test(t + hh/2, yt1); yt += hh/3*k; yt1 = yi + hh*k;
        k = f_test(t + hh, yt1); yt += hh/6*k;
        /* half step 2 — FIXED: save yt2 as starting point */
        real_t yt2 = yt;
        k = f_test(t + hh, yt2); yt = yt2 + hh/6*k; yt1 = yt2 + 0.5*hh*k;
        k = f_test(t + hh + hh/2, yt1); yt += hh/3*k; yt1 = yt2 + 0.5*hh*k;
        k = f_test(t + hh + hh/2, yt1); yt += hh/3*k; yt1 = yt2 + hh*k;
        k = f_test(t + hh + hh, yt1); yt += hh/6*k;
        ye = 8 * 0.5 * (yt - y1) / 15;
        y = yt; t += h;
    }
    result[i] = y; yerrs[i] = ye;
}
#endif

void rk4_gpu(const real_t *y0, int n, real_t h, int steps, real_t *result, double *gpu_time) {
    real_t *d_y0, *d_result;
    cudaMalloc(&d_y0, n*sizeof(real_t)); cudaMalloc(&d_result, n*sizeof(real_t));
    cudaMemcpy(d_y0, y0, n*sizeof(real_t), cudaMemcpyHostToDevice);
    int threads=256, blocks=(n+255)/256;
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); rk4_kernel<<<blocks,threads>>>(d_y0,d_result,h,steps,n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    *gpu_time = (double)gpu_elapsed(s,e);
    cudaMemcpy(result, d_result, n*sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaEventDestroy(s); cudaEventDestroy(e); cudaFree(d_y0); cudaFree(d_result);
}

void rk4sd_gpu(const real_t *y0, int n, real_t h, int steps, real_t *result, real_t *yerrs, double *gpu_time) {
    real_t *d_y0, *d_result, *d_yerrs;
    cudaMalloc(&d_y0, n*sizeof(real_t)); cudaMalloc(&d_result, n*sizeof(real_t)); cudaMalloc(&d_yerrs, n*sizeof(real_t));
    cudaMemcpy(d_y0, y0, n*sizeof(real_t), cudaMemcpyHostToDevice);
    int threads=256, blocks=(n+255)/256;
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); rk4sd_kernel<<<blocks,threads>>>(d_y0,d_result,d_yerrs,h,steps,n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    *gpu_time = (double)gpu_elapsed(s,e);
    cudaMemcpy(result, d_result, n*sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(yerrs, d_yerrs, n*sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaEventDestroy(s); cudaEventDestroy(e); cudaFree(d_y0); cudaFree(d_result); cudaFree(d_yerrs);
}

int main(void) {
    int n = TEST_SIZE; real_t h=0.01; int steps=NSTEPS; real_t T=steps*h;
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("RK4 GPU n=%d GPU=%s", n, p.name);
#ifdef OPT_FP32
    printf(" FP32");
#endif
#ifdef OPT_ALGO
    printf(" +ALGO");
#endif
    printf("\n");
    real_t *y0=(real_t*)malloc(n*sizeof(real_t)),*exact=(real_t*)malloc(n*sizeof(real_t));
    real_t *result=(real_t*)malloc(n*sizeof(real_t)),*yerrs=(real_t*)malloc(n*sizeof(real_t));
    srand(42);
    for(int i=0;i<n;i++) y0[i]=0.5+1.5*(real_t)rand()/RAND_MAX;
    for(int i=0;i<n;i++) exact[i]=exact_sol(T,y0[i]);
    double gt;
    rk4_gpu(y0,n,h,steps,result,&gt);
    printf("  [1] Standard RK4 GPU: %.4f s\n",gt);
    int errs=0;
    for(int i=0;i<n;i++){double d=R_FABS(exact[i]-result[i]);if(d/(R_FABS(exact[i])+1e-15)>1e-4)errs++;}
    printf("  RK4 vs Exact: %s\n",errs==0?"ALL PASS":"FAIL");
    rk4sd_gpu(y0,n,h,steps,result,yerrs,&gt);
    printf("  [2] RK4SD GPU: %.4f s\n",gt);
    errs=0;
    for(int i=0;i<n;i++){double d=R_FABS(exact[i]-result[i]);if(d/(R_FABS(exact[i])+1e-15)>1e-5)errs++;}
    printf("  RK4SD vs Exact: %s\n",errs==0?"ALL PASS":"FAIL");
    free(y0);free(exact);free(result);free(yerrs);
    return 0;
}
