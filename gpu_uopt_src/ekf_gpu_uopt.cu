/* ekf_gpu_uopt.cu — EKF GPU unified optimization (see GPU优化详解.md) */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STEPS 100
#ifndef BATCH
#define BATCH 65536
#endif
#define DIM 4
#define MDIM 2
#define TRUE_ALPHA 9.81
#define TRUE_BETA 0.30
#define DT 0.01

#ifdef OPT_FP32
typedef float real;
#define SINF __sinf
#define COSF __cosf
#define R_ABS fabsf
#define R_CONST(v) v##f
#else
typedef double real;
#define SINF sin
#define COSF cos
#define R_ABS fabs
#define R_CONST(v) v
#endif

static inline float gpu_elapsed(cudaEvent_t s,cudaEvent_t e){float ms;cudaEventElapsedTime(&ms,s,e);return ms/1000.0f;}
static inline real frand(void){return (real)rand()/(real)RAND_MAX;}
static inline real frand_normal(void){real u1=frand()+R_CONST(1e-9),u2=frand();return sqrt(-2*log(u1))*cos(R_CONST(6.283185307)*u2);}

__constant__ real d_Q[16];
__constant__ real d_R[4];
__constant__ real d_dt;

#ifdef OPT_ALGO
__device__ void m4(const real a[16],const real b[16],real c[16]){
    c[0]=a[0]*b[0]+a[1]*b[4]+a[2]*b[8]+a[3]*b[12];c[1]=a[0]*b[1]+a[1]*b[5]+a[2]*b[9]+a[3]*b[13];
    c[2]=a[0]*b[2]+a[1]*b[6]+a[2]*b[10]+a[3]*b[14];c[3]=a[0]*b[3]+a[1]*b[7]+a[2]*b[11]+a[3]*b[15];
    c[4]=a[4]*b[0]+a[5]*b[4]+a[6]*b[8]+a[7]*b[12];c[5]=a[4]*b[1]+a[5]*b[5]+a[6]*b[9]+a[7]*b[13];
    c[6]=a[4]*b[2]+a[5]*b[6]+a[6]*b[10]+a[7]*b[14];c[7]=a[4]*b[3]+a[5]*b[7]+a[6]*b[11]+a[7]*b[15];
    c[8]=a[8]*b[0]+a[9]*b[4]+a[10]*b[8]+a[11]*b[12];c[9]=a[8]*b[1]+a[9]*b[5]+a[10]*b[9]+a[11]*b[13];
    c[10]=a[8]*b[2]+a[9]*b[6]+a[10]*b[10]+a[11]*b[14];c[11]=a[8]*b[3]+a[9]*b[7]+a[10]*b[11]+a[11]*b[15];
    c[12]=a[12]*b[0]+a[13]*b[4]+a[14]*b[8]+a[15]*b[12];c[13]=a[12]*b[1]+a[13]*b[5]+a[14]*b[9]+a[15]*b[13];
    c[14]=a[12]*b[2]+a[13]*b[6]+a[14]*b[10]+a[15]*b[14];c[15]=a[12]*b[3]+a[13]*b[7]+a[14]*b[11]+a[15]*b[15];
}
#else
__device__ void m4(const real a[16],const real b[16],real c[16]){for(int i=0;i<4;i++)for(int j=0;j<4;j++){real s=0;for(int k=0;k<4;k++)s+=a[i*4+k]*b[k*4+j];c[i*4+j]=s;}}
#endif

__device__ void m4AtB(const real a[16],const real b[16],real c[16]){for(int i=0;i<4;i++)for(int j=0;j<4;j++){real s=0;for(int k=0;k<4;k++)s+=a[k*4+i]*b[k*4+j];c[i*4+j]=s;}}

__device__ void ekf_f(const real*x,real dt,real*xo){real th=x[0],om=x[1],al=x[2],be=x[3],st=SINF(th);xo[0]=th+om*dt;xo[1]=om-al*st*dt-be*om*dt;xo[2]=al;xo[3]=be;}
__device__ void ekf_F(const real*x,real dt,real*F){for(int i=0;i<16;i++)F[i]=0;F[0]=1;F[1]=dt;F[4]=-x[2]*COSF(x[0])*dt;F[5]=1-x[3]*dt;F[6]=-SINF(x[0])*dt;F[7]=-x[1]*dt;F[10]=1;F[15]=1;}
__device__ void ekf_H(const real*x,real*H){for(int i=0;i<8;i++)H[i]=0;H[0]=1;H[4]=COSF(x[0]);}

__global__ void ekf_kernel(real*st,real*cv,const real*meas,int steps,int batch){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=batch)return;
    real x[4],P[16];for(int i=0;i<4;i++)x[i]=st[b*4+i];for(int i=0;i<16;i++)P[i]=cv[b*16+i];const real*z=&meas[b*steps*2];
    for(int s=0;s<steps;s++){
        const real*_Q=d_Q;const real*_R=d_R;real _dt=d_dt;
        real xp[4];ekf_f(x,_dt,xp);real F_t[16];ekf_F(x,_dt,F_t);real t1[16];m4(F_t,P,t1);real Pp[16];m4AtB(t1,F_t,Pp);for(int _i=0;_i<16;_i++)Pp[_i]+=_Q[_i];
        real H_t[8];ekf_H(xp,H_t);real HP[8];for(int _i=0;_i<2;_i++)for(int _j=0;_j<4;_j++){real _s=0;for(int _k=0;_k<4;_k++)_s+=H_t[_i*4+_k]*Pp[_k*4+_j];HP[_i*4+_j]=_s;}
        real S_t[4];for(int _i=0;_i<2;_i++)for(int _j=0;_j<2;_j++){real _s=0;for(int _k=0;_k<4;_k++)_s+=HP[_i*4+_k]*H_t[_j*4+_k];S_t[_i*2+_j]=_s+_R[_i*2+_j];}
        real Kt[8];for(int _i=0;_i<4;_i++)for(int _j=0;_j<2;_j++){real _s=0;for(int _k=0;_k<4;_k++)_s+=Pp[_i*4+_k]*H_t[_j*4+_k];Kt[_j*4+_i]=_s;}
        {real _a=S_t[0],_b=S_t[1],_c=S_t[3];real _det=_a*_c-_b*_b;if(R_ABS(_det)<R_CONST(1e-12)){_a+=R_CONST(1e-6);_c+=R_CONST(1e-6);_det=_a*_c-_b*_b;}real _inv=R_CONST(1.0)/_det;real _si00=_c*_inv,_si01=-_b*_inv,_si10=-_b*_inv,_si11=_a*_inv;real _K0[4],_K1[4];for(int _i=0;_i<4;_i++){_K0[_i]=Kt[0*4+_i];_K1[_i]=Kt[1*4+_i];}for(int _i=0;_i<4;_i++){Kt[0*4+_i]=_si00*_K0[_i]+_si01*_K1[_i];Kt[1*4+_i]=_si10*_K0[_i]+_si11*_K1[_i];}}
        real zp[2];zp[0]=xp[0];zp[1]=SINF(xp[0]);real innov[2];innov[0]=z[s*2+0]-zp[0];innov[1]=z[s*2+1]-zp[1];for(int _i=0;_i<4;_i++)x[_i]=xp[_i]+Kt[0*4+_i]*innov[0]+Kt[1*4+_i]*innov[1];
        real KH[16];for(int _i=0;_i<4;_i++)for(int _j=0;_j<4;_j++){real _s=0;for(int _k=0;_k<2;_k++)_s+=Kt[_k*4+_i]*H_t[_k*4+_j];KH[_i*4+_j]=_s;}real id[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};for(int _i=0;_i<16;_i++)id[_i]-=KH[_i];m4(id,Pp,P);
    }
    for(int i=0;i<4;i++)st[b*4+i]=x[i];for(int i=0;i<16;i++)cv[b*16+i]=P[i];
}

static void run_ekf_kernel(real*hs,real*hc,real*hm,const real*Q,const real*RR,real dt,double*gt){
    int batch=(int)BATCH;real*ds,*dc,*dm;
    cudaMalloc(&ds,(size_t)batch*4*sizeof(real));cudaMalloc(&dc,(size_t)batch*16*sizeof(real));cudaMalloc(&dm,(size_t)batch*STEPS*2*sizeof(real));
    cudaMemcpy(ds,hs,(size_t)batch*4*sizeof(real),cudaMemcpyHostToDevice);cudaMemcpy(dc,hc,(size_t)batch*16*sizeof(real),cudaMemcpyHostToDevice);cudaMemcpy(dm,hm,(size_t)batch*STEPS*2*sizeof(real),cudaMemcpyHostToDevice);
    int th=64,bl=(batch+63)/64;cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    (void)Q;(void)RR;(void)dt;
    ekf_kernel<<<bl,th>>>(ds,dc,dm,STEPS,batch);cudaDeviceSynchronize();
    cudaMemcpy(ds,hs,(size_t)batch*4*sizeof(real),cudaMemcpyHostToDevice);cudaMemcpy(dc,hc,(size_t)batch*16*sizeof(real),cudaMemcpyHostToDevice);
    cudaEventRecord(s);ekf_kernel<<<bl,th>>>(ds,dc,dm,STEPS,batch);cudaEventRecord(e);cudaEventSynchronize(e);
    *gt=(double)gpu_elapsed(s,e);
    cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(ds);cudaFree(dc);cudaFree(dm);
}

int main(void){real dt=R_CONST(0.01);cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("EKF GPU dim=%d batch=%d GPU=%s ",DIM,(int)BATCH,p.name);
#ifdef OPT_FP32
    printf("FP32 ");
#else
    printf("FP64 ");
#endif
#ifdef OPT_ALGO
    printf("+ALGO");
#endif
    printf("\n");
    real Q[16]={0};Q[0]=R_CONST(0.001);Q[5]=R_CONST(0.001);Q[10]=R_CONST(1e-5);Q[15]=R_CONST(1e-5);
    real RR[4]={0};RR[0]=R_CONST(0.003);RR[3]=R_CONST(0.003);
    cudaMemcpyToSymbol(d_Q,Q,sizeof(Q));cudaMemcpyToSymbol(d_R,RR,sizeof(RR));cudaMemcpyToSymbol(d_dt,&dt,sizeof(real));
    int batch=(int)BATCH;
    real*hs=(real*)malloc((size_t)batch*4*sizeof(real));real*hc=(real*)malloc((size_t)batch*16*sizeof(real));real*hm=(real*)malloc((size_t)batch*STEPS*2*sizeof(real));
    srand(42);
    for(int b=0;b<batch;b++){
        hs[b*4+0]=R_CONST(1.0)+R_CONST(0.2)*frand_normal();hs[b*4+1]=R_CONST(0.0)+R_CONST(0.1)*frand_normal();hs[b*4+2]=TRUE_ALPHA+R_CONST(2.0)*frand_normal();hs[b*4+3]=TRUE_BETA+R_CONST(0.5)*frand_normal();
        for(int i=0;i<16;i++)hc[b*16+i]=0;hc[b*16+0]=R_CONST(0.5);hc[b*16+5]=R_CONST(0.5);hc[b*16+10]=R_CONST(0.5);hc[b*16+15]=R_CONST(0.5);
        real th=R_CONST(1.0)+R_CONST(0.1)*frand_normal(),om=R_CONST(0.0)+R_CONST(0.05)*frand_normal();
        for(int s=0;s<STEPS;s++){real stt=sin(th);th+=om*dt+R_CONST(0.01)*frand_normal();om+=(-TRUE_ALPHA*stt*dt-TRUE_BETA*om*dt)+R_CONST(0.01)*frand_normal();hm[b*STEPS*2+s*2+0]=th+R_CONST(0.05)*frand_normal();hm[b*STEPS*2+s*2+1]=sin(th)+R_CONST(0.05)*frand_normal();}
    }
    double gt;run_ekf_kernel(hs,hc,hm,Q,RR,dt,&gt);printf("  GPU time: %.6f s\n",gt);
    free(hs);free(hc);free(hm);return 0;
}
