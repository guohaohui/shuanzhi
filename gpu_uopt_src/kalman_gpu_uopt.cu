/* kalman_gpu_uopt.cu — Kalman GPU unified optimization */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef BATCH_SIZE
#define BATCH_SIZE 1024
#endif
#define D 7
#define DS 49

#ifdef OPT_FP32
typedef float real;
#define R_ABS fabsf
#define R_SQRT sqrtf
#define R_CONST(v) v##f
#else
typedef double real;
#define R_ABS fabs
#define R_SQRT sqrt
#define R_CONST(v) v
#endif

static inline float ge(cudaEvent_t s, cudaEvent_t e){float ms;cudaEventElapsedTime(&ms,s,e);return ms/1000.0f;}
static inline real rfr(void){return (real)rand()/(real)RAND_MAX;}
static inline real rfrr(real l,real h){return l+(h-l)*rfr();}

__host__ __device__ static void mid7(real m[DS]){
    for(int i=0;i<DS;i++)m[i]=R_CONST(0.0);
    m[0]=R_CONST(1.0);m[8]=R_CONST(1.0);m[16]=R_CONST(1.0);m[24]=R_CONST(1.0);
    m[32]=R_CONST(1.0);m[40]=R_CONST(1.0);m[48]=R_CONST(1.0);
}

#if defined(OPT_ALGO)
__device__ void m7(const real a[DS],const real b[DS],real c[DS]){
#define R(i,j) a[i]*b[j]+a[i+1]*b[7+j]+a[i+2]*b[14+j]+a[i+3]*b[21+j]+a[i+4]*b[28+j]+a[i+5]*b[35+j]+a[i+6]*b[42+j]
    c[0]=R(0,0);c[1]=R(0,1);c[2]=R(0,2);c[3]=R(0,3);c[4]=R(0,4);c[5]=R(0,5);c[6]=R(0,6);
    c[7]=R(7,0);c[8]=R(7,1);c[9]=R(7,2);c[10]=R(7,3);c[11]=R(7,4);c[12]=R(7,5);c[13]=R(7,6);
    c[14]=R(14,0);c[15]=R(14,1);c[16]=R(14,2);c[17]=R(14,3);c[18]=R(14,4);c[19]=R(14,5);c[20]=R(14,6);
    c[21]=R(21,0);c[22]=R(21,1);c[23]=R(21,2);c[24]=R(21,3);c[25]=R(21,4);c[26]=R(21,5);c[27]=R(21,6);
    c[28]=R(28,0);c[29]=R(28,1);c[30]=R(28,2);c[31]=R(28,3);c[32]=R(28,4);c[33]=R(28,5);c[34]=R(28,6);
    c[35]=R(35,0);c[36]=R(35,1);c[37]=R(35,2);c[38]=R(35,3);c[39]=R(35,4);c[40]=R(35,5);c[41]=R(35,6);
    c[42]=R(42,0);c[43]=R(42,1);c[44]=R(42,2);c[45]=R(42,3);c[46]=R(42,4);c[47]=R(42,5);c[48]=R(42,6);
#undef R
}
__device__ void m1(const real a[DS],const real x[D],real y[D]){
    y[0]=a[0]*x[0]+a[1]*x[1]+a[2]*x[2]+a[3]*x[3]+a[4]*x[4]+a[5]*x[5]+a[6]*x[6];
    y[1]=a[7]*x[0]+a[8]*x[1]+a[9]*x[2]+a[10]*x[3]+a[11]*x[4]+a[12]*x[5]+a[13]*x[6];
    y[2]=a[14]*x[0]+a[15]*x[1]+a[16]*x[2]+a[17]*x[3]+a[18]*x[4]+a[19]*x[5]+a[20]*x[6];
    y[3]=a[21]*x[0]+a[22]*x[1]+a[23]*x[2]+a[24]*x[3]+a[25]*x[4]+a[26]*x[5]+a[27]*x[6];
    y[4]=a[28]*x[0]+a[29]*x[1]+a[30]*x[2]+a[31]*x[3]+a[32]*x[4]+a[33]*x[5]+a[34]*x[6];
    y[5]=a[35]*x[0]+a[36]*x[1]+a[37]*x[2]+a[38]*x[3]+a[39]*x[4]+a[40]*x[5]+a[41]*x[6];
    y[6]=a[42]*x[0]+a[43]*x[1]+a[44]*x[2]+a[45]*x[3]+a[46]*x[4]+a[47]*x[5]+a[48]*x[6];
}
__device__ void mbt(const real a[DS],const real b[DS],real c[DS]){
    for(int i=0;i<D;i++){const real*ar=&a[i*D];for(int j=0;j<D;j++){const real*br=&b[j*D];c[i*D+j]=ar[0]*br[0]+ar[1]*br[1]+ar[2]*br[2]+ar[3]*br[3]+ar[4]*br[4]+ar[5]*br[5]+ar[6]*br[6];}}
}
__device__ int chol7(real A[DS]){
    for(int j=0;j<D;j++){real s=R_CONST(0.0);for(int k=0;k<j;k++){real v=A[j*D+k];s+=v*v;}real d=A[j*D+j]-s;if(d<=R_CONST(1e-12))return 0;A[j*D+j]=R_SQRT(d);for(int i=j+1;i<D;i++){s=R_CONST(0.0);for(int k=0;k<j;k++)s+=A[i*D+k]*A[j*D+k];A[i*D+j]=(A[i*D+j]-s)/A[j*D+j];}}return 1;
}
__device__ void cs7(const real L[DS],real B[DS]){
    for(int c=0;c<D;c++){for(int i=0;i<D;i++){real s=B[i*D+c];for(int k=0;k<i;k++)s-=L[i*D+k]*B[k*D+c];B[i*D+c]=s/L[i*D+i];}}
    for(int c=0;c<D;c++){for(int i=D-1;i>=0;i--){real s=B[i*D+c];for(int k=i+1;k<D;k++)s-=L[k*D+i]*B[k*D+c];B[i*D+c]=s/L[i*D+i];}}
}
#else
__device__ void m7(const real a[DS],const real b[DS],real c[DS]){for(int i=0;i<D;i++)for(int j=0;j<D;j++){real s=R_CONST(0.0);for(int k=0;k<D;k++)s+=a[i*D+k]*b[k*D+j];c[i*D+j]=s;}}
__device__ void m1(const real a[DS],const real x[D],real y[D]){for(int i=0;i<D;i++){real s=R_CONST(0.0);for(int k=0;k<D;k++)s+=a[i*D+k]*x[k];y[i]=s;}}
__device__ void mbt(const real a[DS],const real b[DS],real c[DS]){for(int i=0;i<D;i++)for(int j=0;j<D;j++){real s=R_CONST(0.0);for(int k=0;k<D;k++)s+=a[i*D+k]*b[j*D+k];c[i*D+j]=s;}}
__device__ int gauss7(real A[DS],real B[DS]){
    int pv[D];for(int i=0;i<D;i++)pv[i]=i;
    for(int col=0;col<D;col++){int br=col;real bv=R_ABS(A[pv[col]*D+col]);for(int r=col+1;r<D;r++){real v=R_ABS(A[pv[r]*D+col]);if(v>bv){bv=v;br=r;}}if(bv<R_CONST(1e-15))return 0;int t=pv[col];pv[col]=pv[br];pv[br]=t;real inv=R_CONST(1.0)/A[pv[col]*D+col];for(int j=col;j<D;j++)A[pv[col]*D+j]*=inv;for(int j=0;j<D;j++)B[pv[col]*D+j]*=inv;for(int r=0;r<D;r++){if(r==col)continue;real f=A[pv[r]*D+col];for(int j=col;j<D;j++)A[pv[r]*D+j]-=f*A[pv[col]*D+j];for(int j=0;j<D;j++)B[pv[r]*D+j]-=f*B[pv[col]*D+j];}}real tmp[D];for(int i=0;i<D;i++){if(pv[i]!=i){for(int j=0;j<D;j++){tmp[j]=B[i*D+j];B[i*D+j]=B[pv[i]*D+j];B[pv[i]*D+j]=tmp[j];}for(int k=0;k<D;k++){if(pv[k]==i){pv[k]=pv[i];break;}}pv[i]=i;}}return 1;
}
#endif

__global__ void kalman_kernel(real *ss,real *cs,const real *F,const real *H,const real *Q,const real *R,const real *meas,int dim,int steps,int batch){
    int problem=blockIdx.x*blockDim.x+threadIdx.x;
#ifdef OPT_MEM
    __shared__ real sF[DS],sH[DS],sQ[DS],sR[DS];int tid=threadIdx.x;if(tid<DS){sF[tid]=F[tid];sH[tid]=H[tid];sQ[tid]=Q[tid];sR[tid]=R[tid];}__syncthreads();
#define MF sF
#define MH sH
#define MQ sQ
#define MR sR
#else
#define MF F
#define MH H
#define MQ Q
#define MR R
#endif
    if(problem>=batch)return;
    real st[D],P[DS];
    for(int i=0;i<dim;i++)st[i]=ss[problem*dim+i];
    for(int i=0;i<DS;i++)P[i]=cs[problem*DS+i];
    const real *z=&meas[problem*steps*dim];
    for(int s=0;s<steps;s++){
#if defined(OPT_ALGO)
        real sp[D];m1(MF,st,sp);real tFP[DS];m7(MF,P,tFP);real Pp[DS];mbt(tFP,MF,Pp);for(int i=0;i<DS;i++)Pp[i]+=MQ[i];
        real HP[DS],S_t[DS];m7(MH,Pp,HP);mbt(HP,MH,S_t);for(int i=0;i<DS;i++)S_t[i]+=MR[i];
        real Kt[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)Kt[i*D+j]=HP[j*D+i];
        chol7(S_t);cs7(S_t,Kt);
        real G[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)G[i*D+j]=Kt[j*D+i];
        real Hx[D];m1(MH,sp,Hx);real innov[D];for(int i=0;i<D;i++)innov[i]=z[s*dim+i]-Hx[i];
        real corr[D];m1(G,innov,corr);for(int i=0;i<D;i++)st[i]=sp[i]+corr[i];
        real KH[DS];m7(G,MH,KH);for(int i=0;i<D;i++)for(int j=0;j<D;j++)KH[i*D+j]=(i==j?R_CONST(1.0):R_CONST(0.0))-KH[i*D+j];
        m7(KH,Pp,P);
#else
        real xp[D];m1(MF,st,xp);real FP[DS];m7(MF,P,FP);real Pp[DS];mbt(FP,MF,Pp);for(int i=0;i<DS;i++)Pp[i]+=MQ[i];
        real HP[DS],S_t[DS];m7(MH,Pp,HP);mbt(HP,MH,S_t);for(int i=0;i<DS;i++)S_t[i]+=MR[i];
        real Kt[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)Kt[i*D+j]=HP[j*D+i];
        gauss7(S_t,Kt);
        real G[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)G[i*D+j]=Kt[j*D+i];
        real Hx[D];m1(MH,xp,Hx);real innov[D];for(int i=0;i<D;i++)innov[i]=z[s*dim+i]-Hx[i];
        real corr[D];m1(G,innov,corr);for(int i=0;i<D;i++)st[i]=xp[i]+corr[i];
        real KH[DS];m7(G,MH,KH);for(int i=0;i<D;i++)for(int j=0;j<D;j++)KH[i*D+j]=(i==j?R_CONST(1.0):R_CONST(0.0))-KH[i*D+j];
        m7(KH,Pp,P);
#endif
    }
    for(int i=0;i<dim;i++)ss[problem*dim+i]=st[i];
    for(int i=0;i<DS;i++)cs[problem*DS+i]=P[i];
#undef MF
#undef MH
#undef MQ
#undef MR
}

void kalman_batch_gpu(real *ss,real *cs,const real *F,const real *H,const real *Q,const real *R,const real *meas,int dim,int steps,int batch,double *gt){
    real *ds,*dc,*dm,*dF,*dH,*dQ,*dR;
    size_t sb=(size_t)batch*dim*sizeof(real),cb=(size_t)batch*DS*sizeof(real),mb=(size_t)batch*steps*dim*sizeof(real),matb=(size_t)DS*sizeof(real);
    cudaMalloc(&ds,sb);cudaMalloc(&dc,cb);cudaMalloc(&dm,mb);cudaMalloc(&dF,matb);cudaMalloc(&dH,matb);cudaMalloc(&dQ,matb);cudaMalloc(&dR,matb);
    cudaMemcpy(ds,ss,sb,cudaMemcpyHostToDevice);cudaMemcpy(dc,cs,cb,cudaMemcpyHostToDevice);cudaMemcpy(dm,meas,mb,cudaMemcpyHostToDevice);
    cudaMemcpy(dF,F,matb,cudaMemcpyHostToDevice);cudaMemcpy(dH,H,matb,cudaMemcpyHostToDevice);cudaMemcpy(dQ,Q,matb,cudaMemcpyHostToDevice);cudaMemcpy(dR,R,matb,cudaMemcpyHostToDevice);
    int th=128,bl=(batch+th-1)/th;
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    cudaEventRecord(s);kalman_kernel<<<bl,th>>>(ds,dc,dF,dH,dQ,dR,dm,dim,steps,batch);cudaEventRecord(e);cudaEventSynchronize(e);
    *gt=(double)ge(s,e);
    cudaMemcpy(ss,ds,sb,cudaMemcpyDeviceToHost);cudaMemcpy(cs,dc,cb,cudaMemcpyDeviceToHost);
    cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(ds);cudaFree(dc);cudaFree(dm);cudaFree(dF);cudaFree(dH);cudaFree(dQ);cudaFree(dR);
}

int main(void){
    int dim=D,steps=100,batch=BATCH_SIZE;
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("Kalman GPU dim=%d batch=%d GPU=%s",dim,batch,p.name);
#ifdef OPT_FP32
    printf(" float");
#else
    printf(" double");
#endif
#ifdef OPT_ALGO
    printf("+Algo");
#endif
#ifdef OPT_MEM
    printf("+SharedMem");
#endif
    printf("\n");
    real F[DS],H[DS],Q[DS],R[DS],Rn[DS];
    mid7(F);mid7(H);mid7(Q);
    for(int i=0;i<DS;i++)R[i]=R_CONST(0.0);
    mid7(Rn);real sigma=R_CONST(1.0);for(int i=0;i<DS;i++)Rn[i]*=sigma*sigma;
    printf("--- Test1: No noise ---\n");
    {srand(42);real st[D],se[D],P[DS],meas[700];
    for(int i=0;i<dim;i++)st[i]=rfrr(-R_CONST(1.0),R_CONST(1.0));
    for(int i=0;i<dim;i++)se[i]=R_CONST(0.0);mid7(P);
    for(int s=0;s<steps;s++)for(int i=0;i<dim;i++)meas[s*dim+i]=st[i];
    double gt;kalman_batch_gpu(se,P,F,H,Q,R,meas,dim,steps,1,&gt);
    real me=R_CONST(0.0);for(int i=0;i<dim;i++){real e=R_ABS(se[i]-st[i]);if(e>me)me=e;}
    printf("  Max error: %e %s\n",(double)me,me<R_CONST(1e-4)?"PASS":"FAIL");}
    printf("--- Test2: With noise ---\n");
    {srand(42);real st[D],se[D],P[DS],meas[700];
    for(int i=0;i<dim;i++)st[i]=rfrr(-R_CONST(1.0),R_CONST(1.0));
    for(int i=0;i<dim;i++)se[i]=R_CONST(0.0);mid7(P);
    real ns=R_CONST(0.0);
    for(int s=0;s<steps;s++){for(int i=0;i<dim;i++){real n=rfrr(-R_CONST(1.0),R_CONST(1.0))*sigma;if(s==steps-1)ns+=R_ABS(n);meas[s*dim+i]=st[i]+n;}}
    ns/=(real)dim;double gt;kalman_batch_gpu(se,P,F,H,Q,Rn,meas,dim,steps,1,&gt);
    real err=R_CONST(0.0);for(int i=0;i<dim;i++)err+=R_ABS(se[i]-st[i]);err/=(real)dim;
    printf("  Est error: %e, noise: %.4f %s\n",(double)err,(double)ns,err<ns?"PASS":"FAIL");}
    printf("--- Test3: Batch perf ---\n");
    {real*ss=(real*)malloc((size_t)batch*dim*sizeof(real));real*cs_mat=(real*)malloc((size_t)batch*DS*sizeof(real));real*ms=(real*)malloc((size_t)batch*steps*dim*sizeof(real));
    srand(42);for(int b=0;b<batch;b++){for(int i=0;i<dim;i++)ss[b*dim+i]=R_CONST(0.0);mid7(&cs_mat[b*DS]);for(int s=0;s<steps;s++)for(int i=0;i<dim;i++)ms[b*steps*dim+s*dim+i]=rfrr(-R_CONST(1.0),R_CONST(1.0))+rfrr(-R_CONST(1.0),R_CONST(1.0))*sigma;}
    double gt;kalman_batch_gpu(ss,cs_mat,F,H,Q,Rn,ms,dim,steps,batch,&gt);printf("  GPU time: %.4f s\n",gt);
    free(ss);free(cs_mat);free(ms);}
    return 0;
}
