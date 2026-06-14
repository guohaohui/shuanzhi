/* ekf_gpu_base.cu — EKF GPU baseline (double, generic loops, kernel-param Q/R/dt) */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef BATCH
#define BATCH 1024
#endif
#define TA 9.81
#define TB 0.30
#define DIM 4
#define MDIM 2

static inline float gpu_elapsed(cudaEvent_t s,cudaEvent_t e){float ms;cudaEventElapsedTime(&ms,s,e);return ms/1000.0f;}

__device__ void m4(const double a[16],const double b[16],double c[16]){for(int i=0;i<4;i++)for(int j=0;j<4;j++){double s=0;for(int k=0;k<4;k++)s+=a[i*4+k]*b[k*4+j];c[i*4+j]=s;}}
__device__ void m4AtB(const double a[16],const double b[16],double c[16]){for(int i=0;i<4;i++)for(int j=0;j<4;j++){double s=0;for(int k=0;k<4;k++)s+=a[k*4+i]*b[k*4+j];c[i*4+j]=s;}}

__device__ void ekf_f(const double*x,double dt,double*xo){double th=x[0],om=x[1],al=x[2],be=x[3],st=sin(th);xo[0]=th+om*dt;xo[1]=om-al*st*dt-be*om*dt;xo[2]=al;xo[3]=be;}
__device__ void ekf_F(const double*x,double dt,double*F){for(int i=0;i<16;i++)F[i]=0;F[0]=1;F[1]=dt;F[4]=-x[2]*cos(x[0])*dt;F[5]=1-x[3]*dt;F[6]=-sin(x[0])*dt;F[7]=-x[1]*dt;F[10]=1;F[15]=1;}
__device__ void ekf_H(const double*x,double*H){for(int i=0;i<8;i++)H[i]=0;H[0]=1;H[4]=cos(x[0]);}

__global__ void ekf_kernel(double*st,double*cv,const double*meas,const double*Q,const double*R,double dt,int steps,int batch){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=batch)return;
    double x[4],P[16];for(int i=0;i<4;i++)x[i]=st[b*4+i];for(int i=0;i<16;i++)P[i]=cv[b*16+i];const double*z=&meas[b*steps*2];
    for(int s=0;s<steps;s++){
        double xp[4];ekf_f(x,dt,xp);double F[16];ekf_F(x,dt,F);double t1[16];m4(F,P,t1);double Pp[16];m4AtB(t1,F,Pp);for(int i=0;i<16;i++)Pp[i]+=Q[i];
        double H[8];ekf_H(xp,H);double HP[8];for(int i=0;i<2;i++)for(int j=0;j<4;j++){double sum=0;for(int k=0;k<4;k++)sum+=H[i*4+k]*Pp[k*4+j];HP[i*4+j]=sum;}
        double S[4];for(int i=0;i<2;i++)for(int j=0;j<2;j++){double sum=0;for(int k=0;k<4;k++)sum+=HP[i*4+k]*H[j*4+k];S[i*2+j]=sum+R[i*2+j];}
        double Kt[8];for(int i=0;i<4;i++)for(int j=0;j<2;j++){double sum=0;for(int k=0;k<4;k++)sum+=Pp[i*4+k]*H[j*4+k];Kt[j*4+i]=sum;}
        {double a=S[0],b_=S[1],c=S[3];double det=a*c-b_*b_;if(fabs(det)<1e-12){a+=1e-6;c+=1e-6;det=a*c-b_*b_;}double inv=1.0/det;double si00=c*inv,si01=-b_*inv,si10=-b_*inv,si11=a*inv;double K0[4],K1[4];for(int i=0;i<4;i++){K0[i]=Kt[0*4+i];K1[i]=Kt[1*4+i];}for(int i=0;i<4;i++){Kt[0*4+i]=si00*K0[i]+si01*K1[i];Kt[1*4+i]=si10*K0[i]+si11*K1[i];}}
        double zp[2];zp[0]=xp[0];zp[1]=sin(xp[0]);double innov[2];innov[0]=z[s*2+0]-zp[0];innov[1]=z[s*2+1]-zp[1];for(int i=0;i<4;i++)x[i]=xp[i]+Kt[0*4+i]*innov[0]+Kt[1*4+i]*innov[1];
        double KH[16];for(int i=0;i<4;i++)for(int j=0;j<4;j++){double sum=0;for(int k=0;k<2;k++)sum+=Kt[k*4+i]*H[k*4+j];KH[i*4+j]=sum;}double id[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};for(int i=0;i<16;i++)id[i]-=KH[i];m4(id,Pp,P);
    }
    for(int i=0;i<4;i++)st[b*4+i]=x[i];for(int i=0;i<16;i++)cv[b*16+i]=P[i];
}

static void run_ekf_kernel(double*hs,double*hc,double*hm,const double*Q,const double*R,double dt,double*gt){
    int batch=(int)BATCH;double*ds,*dc,*dm;
    cudaMalloc(&ds,(size_t)batch*4*sizeof(double));cudaMalloc(&dc,(size_t)batch*16*sizeof(double));cudaMalloc(&dm,(size_t)batch*200*sizeof(double));
    cudaMemcpy(ds,hs,(size_t)batch*4*sizeof(double),cudaMemcpyHostToDevice);cudaMemcpy(dc,hc,(size_t)batch*16*sizeof(double),cudaMemcpyHostToDevice);cudaMemcpy(dm,hm,(size_t)batch*200*sizeof(double),cudaMemcpyHostToDevice);
    int th=64,bl=(batch+63)/64;cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    ekf_kernel<<<bl,th>>>(ds,dc,dm,Q,R,dt,100,batch);cudaDeviceSynchronize();
    cudaMemcpy(ds,hs,(size_t)batch*4*sizeof(double),cudaMemcpyHostToDevice);cudaMemcpy(dc,hc,(size_t)batch*16*sizeof(double),cudaMemcpyHostToDevice);
    cudaEventRecord(s);ekf_kernel<<<bl,th>>>(ds,dc,dm,Q,R,dt,100,batch);cudaEventRecord(e);cudaEventSynchronize(e);
    *gt=(double)gpu_elapsed(s,e);
    cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(ds);cudaFree(dc);cudaFree(dm);
}

int main(void){double dt=0.01;
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("EKF GPU BASE dim=%d batch=%d GPU=%s\n",DIM,(int)BATCH,p.name);
    double Q[16]={0};Q[0]=0.001;Q[5]=0.001;Q[10]=1e-5;Q[15]=1e-5;
    double R[4]={0};R[0]=0.003;R[3]=0.003;
    int batch=(int)BATCH;
    double*hs=(double*)malloc((size_t)batch*4*sizeof(double));double*hc=(double*)malloc((size_t)batch*16*sizeof(double));double*hm=(double*)malloc((size_t)batch*200*sizeof(double));
    srand(42);
    for(int b=0;b<batch;b++){
        hs[b*4+0]=1.0+0.2*((double)rand()/RAND_MAX-0.5);hs[b*4+1]=0.0+0.1*((double)rand()/RAND_MAX-0.5);hs[b*4+2]=TA+2.0*((double)rand()/RAND_MAX-0.5);hs[b*4+3]=TB+0.5*((double)rand()/RAND_MAX-0.5);
        for(int i=0;i<16;i++)hc[b*16+i]=0;hc[b*16+0]=0.5;hc[b*16+5]=0.5;hc[b*16+10]=0.5;hc[b*16+15]=0.5;
        double th=1.0+0.1*((double)rand()/RAND_MAX-0.5),om=0.0+0.05*((double)rand()/RAND_MAX-0.5);
        for(int s=0;s<100;s++){double stt=sin(th);th+=om*dt+0.01*((double)rand()/RAND_MAX-0.5);om+=(-TA*stt*dt-TB*om*dt)+0.01*((double)rand()/RAND_MAX-0.5);hm[b*200+s*2+0]=th+0.05*((double)rand()/RAND_MAX-0.5);hm[b*200+s*2+1]=sin(th)+0.05*((double)rand()/RAND_MAX-0.5);}
    }
    double gt;run_ekf_kernel(hs,hc,hm,Q,R,dt,&gt);printf("  GPU time: %.6f s\n",gt);
    free(hs);free(hc);free(hm);return 0;
}
