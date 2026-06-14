/* rk4_gsl_demo.cu — baseline RK4 demo (CPU/GPU dual-mode), see shuanzhi-完整技术文档.md */
#include "include/common.h"
#ifdef __CUDACC__
__device__ __host__
#endif
static inline double f_test(double t,double y){(void)t;return -y;}
#ifdef __CUDACC__
__device__ __host__
#endif
static inline float exact_solution(float t,float y0){return y0*expf(-t);}
#ifndef __CUDACC__
#define __global__
#define __device__
#define __shared__
#endif
#ifdef __CUDACC__
__device__ __host__
#endif
static inline void rk4_step(double t,double h,double*y,double(*f)(double,double)){double y0=*y;double k=f(t,y0);*y=y0+h/6.0*k;double ytmp=y0+0.5*h*k;k=f(t+0.5*h,ytmp);*y+=h/3.0*k;ytmp=y0+0.5*h*k;k=f(t+0.5*h,ytmp);*y+=h/3.0*k;ytmp=y0+h*k;k=f(t+h,ytmp);*y+=h/6.0*k;}
#ifdef __CUDACC__
__device__ __host__
#endif
static inline void rk4sd_step(double t,double h,double*y,double(*f)(double,double),double*yerr){double y_init=*y;double y_onestep=rk4_step(t,h,y_init,f);double y_twostep=rk4_step(t,h/2.0,y_init,f);y_twostep=rk4_step(t+h/2.0,h/2.0,y_twostep,f);*yerr=8.0*0.5*(y_twostep-y_onestep)/15.0;*y=y_twostep;}
#ifndef USE_CUDA
void rk4_cpu(const float*y0,int n,float h,int steps,float*result){for(int i=0;i<n;i++){double y=(double)y0[i];double t=0.0;for(int s=0;s<steps;s++){rk4_step(t,(double)h,&y,f_test);t+=h;}result[i]=(float)y;}}
void rk4sd_cpu(const float*y0,int n,float h,int steps,float*result,float*yerrs){for(int i=0;i<n;i++){double y=(double)y0[i];double t=0.0;double yerr=0.0;for(int s=0;s<steps;s++){rk4sd_step(t,(double)h,&y,f_test,&yerr);t+=h;}result[i]=(float)y;yerrs[i]=(float)yerr;}}
#endif
#ifdef __CUDACC__
__global__ void rk4_gpu_kernel(const float*y0,float*result,float h,int steps,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double y=(double)y0[i];double t=0.0;for(int s=0;s<steps;s++){rk4_step(t,(double)h,&y,f_test);t+=h;}result[i]=(float)y;}
__global__ void rk4sd_gpu_kernel(const float*y0,float*result,float*yerrs,float h,int steps,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double y=(double)y0[i];double t=0.0;double yerr=0.0;for(int s=0;s<steps;s++){rk4sd_step(t,(double)h,&y,f_test,&yerr);t+=h;}result[i]=(float)y;yerrs[i]=(float)yerr;}
void rk4_gpu(const float*y0,int n,float h,int steps,float*result,double*gpu_time){float*d_y0,*d_result;cudaMalloc(&d_y0,n*sizeof(float));cudaMalloc(&d_result,n*sizeof(float));cudaMemcpy(d_y0,y0,n*sizeof(float),cudaMemcpyHostToDevice);int threads=256,blocks=(n+threads-1)/threads;cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start);rk4_gpu_kernel<<<blocks,threads>>>(d_y0,d_result,h,steps,n);cudaEventRecord(stop);cudaEventSynchronize(stop);*gpu_time=gpu_elapsed(start,stop);cudaMemcpy(result,d_result,n*sizeof(float),cudaMemcpyDeviceToHost);cudaEventDestroy(start);cudaEventDestroy(stop);cudaFree(d_y0);cudaFree(d_result);}
void rk4sd_gpu(const float*y0,int n,float h,int steps,float*result,float*yerrs,double*gpu_time){float*d_y0,*d_result,*d_yerrs;cudaMalloc(&d_y0,n*sizeof(float));cudaMalloc(&d_result,n*sizeof(float));cudaMalloc(&d_yerrs,n*sizeof(float));cudaMemcpy(d_y0,y0,n*sizeof(float),cudaMemcpyHostToDevice);int threads=256,blocks=(n+threads-1)/threads;cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start);rk4sd_gpu_kernel<<<blocks,threads>>>(d_y0,d_result,d_yerrs,h,steps,n);cudaEventRecord(stop);cudaEventSynchronize(stop);*gpu_time=gpu_elapsed(start,stop);cudaMemcpy(result,d_result,n*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(yerrs,d_yerrs,n*sizeof(float),cudaMemcpyDeviceToHost);cudaEventDestroy(start);cudaEventDestroy(stop);cudaFree(d_y0);cudaFree(d_result);cudaFree(d_yerrs);}
#endif
int main(void){int n=TEST_SIZE;float h=0.01f;int steps=NSTEPS;float T=steps*h;printf("RK4 Baseline n=%d GPU=%s\n",n,
#ifdef __CUDACC__
"CUDA"
#else
"CPU"
#endif
);float*y0=(float*)malloc(n*sizeof(float)),*exact=(float*)malloc(n*sizeof(float)),*rk4_result=(float*)malloc(n*sizeof(float)),*sd_result=(float*)malloc(n*sizeof(float)),*sd_yerrs=(float*)malloc(n*sizeof(float)),*gpu_buf=(float*)malloc(n*sizeof(float)),*gpu_yerrs=(float*)malloc(n*sizeof(float));srand(42);for(int i=0;i<n;i++)y0[i]=frand_range(0.5f,2.0f);for(int i=0;i<n;i++)exact[i]=exact_solution(T,y0[i]);
#ifndef __CUDACC__
{double t0=cpu_time();rk4_cpu(y0,n,h,steps,rk4_result);printf("  CPU RK4: %.4f s\n",cpu_time()-t0);}
#else
rk4_cpu(y0,n,h,steps,rk4_result);
#endif
VERIFY("RK4 vs Exact",exact,rk4_result,1e-4,n);
#ifdef __CUDACC__
double gpu_sec;rk4_gpu(y0,n,h,steps,gpu_buf,&gpu_sec);printf("  GPU RK4: %.4f s\n",gpu_sec);VERIFY("RK4 CPU vs GPU",rk4_result,gpu_buf,1e-5,n);
#endif
#ifndef __CUDACC__
{double t0=cpu_time();rk4sd_cpu(y0,n,h,steps,sd_result,sd_yerrs);printf("  CPU RK4SD: %.4f s\n",cpu_time()-t0);}
#else
rk4sd_cpu(y0,n,h,steps,sd_result,sd_yerrs);
#endif
VERIFY("RK4SD vs Exact",exact,sd_result,1e-5,n);
#ifdef __CUDACC__
rk4sd_gpu(y0,n,h,steps,gpu_buf,gpu_yerrs,&gpu_sec);printf("  GPU RK4SD: %.4f s\n",gpu_sec);VERIFY("RK4SD CPU vs GPU",sd_result,gpu_buf,1e-5,n);
#endif
free(y0);free(exact);free(rk4_result);free(sd_result);free(sd_yerrs);free(gpu_buf);free(gpu_yerrs);return 0;}
