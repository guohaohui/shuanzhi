/* common.h — shared header for legacy shuanzhi_cpu_opt (see shuanzhi-完整技术文档.md) */
#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__)||defined(__AVX__)
#include <immintrin.h>
#define HAS_AVX 1
#else
#define HAS_AVX 0
#endif
#if defined(__FMA__)||defined(__AVX2__)
#define HAS_FMA 1
#else
#define HAS_FMA 0
#endif
#if defined(__aarch64__)||defined(__ARM_NEON)||defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif
#ifndef TEST_SIZE
#define TEST_SIZE 65536
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 1024
#endif
#ifndef M_SIZE
#define M_SIZE 8
#endif
#define TRUE_ALPHA 9.81f
#define TRUE_BETA 0.30f
#ifdef _OPENMP
#define CPU_TIME() omp_get_wtime()
#else
static inline double cpu_time(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return(double)ts.tv_sec+(double)ts.tv_nsec*1e-9;}
#define CPU_TIME() cpu_time()
#endif
static inline float frand(void){return(float)rand()/(float)RAND_MAX;}
static inline float frand_range(float lo,float hi){return lo+(hi-lo)*frand();}
static inline float frand_normal(void){float u1=frand()+1e-9f,u2=frand();return sqrtf(-2.0f*logf(u1))*cosf(6.283185307f*u2);}
#endif
