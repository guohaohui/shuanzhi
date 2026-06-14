#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#ifndef TEST_SIZE
#define TEST_SIZE 1024
#endif
#ifndef NSTEPS
#define NSTEPS 100
#endif
static inline double cpu_time(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return(double)ts.tv_sec+(double)ts.tv_nsec*1e-9;}
#ifdef __CUDACC__
static inline float gpu_elapsed(cudaEvent_t start,cudaEvent_t stop){float ms=0;cudaEventElapsedTime(&ms,start,stop);return ms/1000.0f;}
#endif
#define VERIFY(desc,expected,actual,tol,N) do{int _errs=0;for(int _i=0;_i<(N);_i++){double _diff=fabs((double)(expected)[_i]-(double)(actual)[_i]);double _rel=_diff/(fabs((double)(expected)[_i])+1e-15);if(_rel>(tol)){if(_errs<5)printf("  FAIL[%d]: expect %.6f got %.6f (rel=%.2e)\n",_i,(double)(expected)[_i],(double)(actual)[_i],_rel);_errs++;}}if(_errs==0)printf("  OK %s: ALL PASS (tol=%.0e)\n",(desc),(double)(tol));else printf("  ERR %s: %d/%d errors\n",(desc),_errs,(N));}while(0)
static inline float frand(void){return(float)rand()/(float)RAND_MAX;}
static inline float frand_range(float lo,float hi){return lo+(hi-lo)*frand();}
#endif
