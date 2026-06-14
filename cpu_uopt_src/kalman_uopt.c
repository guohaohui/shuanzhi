/* kalman_uopt.c — Kalman CPU unified optimization (8-level) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 1024
#endif
#define D 7
#define DS 49

#ifdef OPT_FLOAT
typedef float real_t;
#define R_ABS fabsf
#define R_SQRT sqrtf
#define R_CONST(v) v##f
#else
typedef double real_t;
#define R_ABS fabs
#define R_SQRT sqrt
#define R_CONST(v) v
#endif

static inline double cpu_time(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return(double)ts.tv_sec+(double)ts.tv_nsec*1e-9;}
static inline real_t rfr(void){return(real_t)rand()/(real_t)RAND_MAX;}
static inline real_t rfrr(real_t l,real_t h){return l+(h-l)*rfr();}
static void mid7(real_t m[DS]){for(int i=0;i<DS;i++)m[i]=R_CONST(0.0);m[0]=R_CONST(1.0);m[8]=R_CONST(1.0);m[16]=R_CONST(1.0);m[24]=R_CONST(1.0);m[32]=R_CONST(1.0);m[40]=R_CONST(1.0);m[48]=R_CONST(1.0);}

#if defined(OPT_ALGO)
static void m7(const real_t a[DS],const real_t b[DS],real_t c[DS]){
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
static void m1(const real_t a[DS],const real_t x[D],real_t y[D]){
    y[0]=a[0]*x[0]+a[1]*x[1]+a[2]*x[2]+a[3]*x[3]+a[4]*x[4]+a[5]*x[5]+a[6]*x[6];
    y[1]=a[7]*x[0]+a[8]*x[1]+a[9]*x[2]+a[10]*x[3]+a[11]*x[4]+a[12]*x[5]+a[13]*x[6];
    y[2]=a[14]*x[0]+a[15]*x[1]+a[16]*x[2]+a[17]*x[3]+a[18]*x[4]+a[19]*x[5]+a[20]*x[6];
    y[3]=a[21]*x[0]+a[22]*x[1]+a[23]*x[2]+a[24]*x[3]+a[25]*x[4]+a[26]*x[5]+a[27]*x[6];
    y[4]=a[28]*x[0]+a[29]*x[1]+a[30]*x[2]+a[31]*x[3]+a[32]*x[4]+a[33]*x[5]+a[34]*x[6];
    y[5]=a[35]*x[0]+a[36]*x[1]+a[37]*x[2]+a[38]*x[3]+a[39]*x[4]+a[40]*x[5]+a[41]*x[6];
    y[6]=a[42]*x[0]+a[43]*x[1]+a[44]*x[2]+a[45]*x[3]+a[46]*x[4]+a[47]*x[5]+a[48]*x[6];
}
static void mbt(const real_t a[DS],const real_t b[DS],real_t c[DS]){for(int i=0;i<D;i++){const real_t*ar=&a[i*D];for(int j=0;j<D;j++){const real_t*br=&b[j*D];c[i*D+j]=ar[0]*br[0]+ar[1]*br[1]+ar[2]*br[2]+ar[3]*br[3]+ar[4]*br[4]+ar[5]*br[5]+ar[6]*br[6];}}}
static int chol7(real_t A[DS]){for(int j=0;j<D;j++){real_t s=R_CONST(0.0);for(int k=0;k<j;k++){real_t v=A[j*D+k];s+=v*v;}real_t d=A[j*D+j]-s;if(d<=R_CONST(1e-12))return 0;A[j*D+j]=R_SQRT(d);for(int i=j+1;i<D;i++){s=R_CONST(0.0);for(int k=0;k<j;k++)s+=A[i*D+k]*A[j*D+k];A[i*D+j]=(A[i*D+j]-s)/A[j*D+j];}}return 1;}
static void cs7(const real_t L[DS],real_t B[DS]){for(int c=0;c<D;c++){for(int i=0;i<D;i++){real_t s=B[i*D+c];for(int k=0;k<i;k++)s-=L[i*D+k]*B[k*D+c];B[i*D+c]=s/L[i*D+i];}}for(int c=0;c<D;c++){for(int i=D-1;i>=0;i--){real_t s=B[i*D+c];for(int k=i+1;k<D;k++)s-=L[k*D+i]*B[k*D+c];B[i*D+c]=s/L[i*D+i];}}}
#else
static void m7(const real_t a[DS],const real_t b[DS],real_t c[DS]){for(int i=0;i<D;i++)for(int j=0;j<D;j++){real_t s=R_CONST(0.0);for(int k=0;k<D;k++)s+=a[i*D+k]*b[k*D+j];c[i*D+j]=s;}}
static void m1(const real_t a[DS],const real_t x[D],real_t y[D]){for(int i=0;i<D;i++){real_t s=R_CONST(0.0);for(int k=0;k<D;k++)s+=a[i*D+k]*x[k];y[i]=s;}}
static void mbt(const real_t a[DS],const real_t b[DS],real_t c[DS]){for(int i=0;i<D;i++)for(int j=0;j<D;j++){real_t s=R_CONST(0.0);for(int k=0;k<D;k++)s+=a[i*D+k]*b[j*D+k];c[i*D+j]=s;}}
static int gauss7(real_t A[DS],real_t B[DS]){int pv[D];for(int i=0;i<D;i++)pv[i]=i;
for(int col=0;col<D;col++){int br=col;real_t bv=R_ABS(A[pv[col]*D+col]);for(int r=col+1;r<D;r++){real_t v=R_ABS(A[pv[r]*D+col]);if(v>bv){bv=v;br=r;}}if(bv<R_CONST(1e-15))return 0;int t=pv[col];pv[col]=pv[br];pv[br]=t;real_t inv=R_CONST(1.0)/A[pv[col]*D+col];for(int j=col;j<D;j++)A[pv[col]*D+j]*=inv;for(int j=0;j<D;j++)B[pv[col]*D+j]*=inv;for(int r=0;r<D;r++){if(r==col)continue;real_t f=A[pv[r]*D+col];for(int j=col;j<D;j++)A[pv[r]*D+j]-=f*A[pv[col]*D+j];for(int j=0;j<D;j++)B[pv[r]*D+j]-=f*B[pv[col]*D+j];}}real_t tmp[D];for(int i=0;i<D;i++){if(pv[i]!=i){for(int j=0;j<D;j++){tmp[j]=B[i*D+j];B[i*D+j]=B[pv[i]*D+j];B[pv[i]*D+j]=tmp[j];}for(int k=0;k<D;k++){if(pv[k]==i){pv[k]=pv[i];break;}}pv[i]=i;}}return 1;}
#endif

int main(void){int dim=D,steps=100,batch=BATCH_SIZE;
    printf("Kalman CPU dim=%d batch=%d",dim,batch);
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
    real_t F[DS],H[DS],Q[DS],R[DS],Rn[DS];
    mid7(F);mid7(H);mid7(Q);
    for(int i=0;i<DS;i++)R[i]=R_CONST(0.0);
    mid7(Rn);real_t sigma=R_CONST(1.0);for(int i=0;i<DS;i++)Rn[i]*=sigma*sigma;

    printf("--- Test1: No noise ---\n");
    {srand(42);real_t st[D],se[D],P[DS];
    for(int i=0;i<dim;i++)st[i]=rfrr(-R_CONST(1.0),R_CONST(1.0));
    for(int i=0;i<dim;i++)se[i]=R_CONST(0.0);mid7(P);
    double t0=cpu_time();
    for(int s=0;s<steps;s++){
        real_t sp[D];m1(F,se,sp);
        real_t tFP[DS];m7(F,P,tFP);real_t Pp[DS];mbt(tFP,F,Pp);for(int i=0;i<DS;i++)Pp[i]+=Q[i];
        real_t HP[DS],St[DS];m7(H,Pp,HP);mbt(HP,H,St);for(int i=0;i<DS;i++)St[i]+=R[i];
        real_t Kt[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)Kt[i*D+j]=HP[j*D+i];
#if defined(OPT_ALGO)
        chol7(St);cs7(St,Kt);
#else
        gauss7(St,Kt);
#endif
        real_t G[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)G[i*D+j]=Kt[j*D+i];
        real_t Hx[D];m1(H,sp,Hx);real_t innov[D];for(int i=0;i<D;i++)innov[i]=st[i]-Hx[i];
        real_t corr[D];m1(G,innov,corr);for(int i=0;i<D;i++)se[i]=sp[i]+corr[i];
        real_t KH[DS];m7(G,H,KH);for(int i=0;i<D;i++)for(int j=0;j<D;j++)KH[i*D+j]=(i==j?R_CONST(1.0):R_CONST(0.0))-KH[i*D+j];m7(KH,Pp,P);
    }
    printf("  Time: %.4f s\n",cpu_time()-t0);
    real_t me=R_CONST(0.0);for(int i=0;i<dim;i++){real_t e=R_ABS(se[i]-st[i]);if(e>me)me=e;}
    printf("  Max error: %e %s\n",(double)me,me<R_CONST(1e-4)?"PASS":"FAIL");}

    printf("--- Test2: With noise ---\n");
    {srand(42);real_t st[D],se[D],P[DS];
    for(int i=0;i<dim;i++)st[i]=rfrr(-R_CONST(1.0),R_CONST(1.0));
    for(int i=0;i<dim;i++)se[i]=R_CONST(0.0);mid7(P);
    real_t ns=R_CONST(0.0);
    double t0=cpu_time();
    for(int s=0;s<steps;s++){
        real_t sp[D];m1(F,se,sp);real_t tFP[DS];m7(F,P,tFP);real_t Pp[DS];mbt(tFP,F,Pp);for(int i=0;i<DS;i++)Pp[i]+=Q[i];
        real_t HP[DS],St[DS];m7(H,Pp,HP);mbt(HP,H,St);for(int i=0;i<DS;i++)St[i]+=Rn[i];
        real_t Kt[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)Kt[i*D+j]=HP[j*D+i];
#if defined(OPT_ALGO)
        chol7(St);cs7(St,Kt);
#else
        gauss7(St,Kt);
#endif
        real_t G[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)G[i*D+j]=Kt[j*D+i];
        real_t meas[D];for(int i=0;i<dim;i++){real_t n=rfrr(-R_CONST(1.0),R_CONST(1.0))*sigma;if(s==steps-1)ns+=R_ABS(n);meas[i]=st[i]+n;}
        real_t Hx[D];m1(H,sp,Hx);real_t innov[D];for(int i=0;i<D;i++)innov[i]=meas[i]-Hx[i];
        real_t corr[D];m1(G,innov,corr);for(int i=0;i<D;i++)se[i]=sp[i]+corr[i];
        real_t KH[DS];m7(G,H,KH);for(int i=0;i<D;i++)for(int j=0;j<D;j++)KH[i*D+j]=(i==j?R_CONST(1.0):R_CONST(0.0))-KH[i*D+j];m7(KH,Pp,P);
    }
    printf("  Time: %.4f s\n",cpu_time()-t0);ns/=(real_t)dim;
    real_t err=R_CONST(0.0);for(int i=0;i<dim;i++)err+=R_ABS(se[i]-st[i]);err/=(real_t)dim;
    printf("  Est error: %e < noise: %.4f %s\n",(double)err,(double)ns,err<ns?"PASS":"FAIL");}

    printf("--- Test3: Batch perf ---\n");
    {real_t*ss=(real_t*)malloc((size_t)batch*dim*sizeof(real_t));real_t*cs_mat=(real_t*)malloc((size_t)batch*DS*sizeof(real_t));real_t*ms=(real_t*)malloc((size_t)batch*steps*dim*sizeof(real_t));
    srand(42);for(int b=0;b<batch;b++){for(int i=0;i<dim;i++)ss[b*dim+i]=R_CONST(0.0);mid7(&cs_mat[b*DS]);for(int s=0;s<steps;s++)for(int i=0;i<dim;i++)ms[b*steps*dim+s*dim+i]=rfrr(-R_CONST(1.0),R_CONST(1.0))+rfrr(-R_CONST(1.0),R_CONST(1.0))*sigma;}
    double t0=cpu_time();
#ifdef OPT_OPENMP
#pragma omp parallel for
#endif
    for(int b=0;b<batch;b++){
        real_t st[D],P[DS];for(int i=0;i<dim;i++)st[i]=ss[b*dim+i];for(int i=0;i<DS;i++)P[i]=cs_mat[b*DS+i];
        const real_t*z=&ms[b*steps*dim];
        for(int s=0;s<steps;s++){
            real_t sp[D];m1(F,st,sp);real_t tFP[DS];m7(F,P,tFP);real_t Pp[DS];mbt(tFP,F,Pp);for(int i=0;i<DS;i++)Pp[i]+=Q[i];
            real_t HP[DS],St[DS];m7(H,Pp,HP);mbt(HP,H,St);for(int i=0;i<DS;i++)St[i]+=Rn[i];
            real_t Kt[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)Kt[i*D+j]=HP[j*D+i];
#if defined(OPT_ALGO)
            chol7(St);cs7(St,Kt);
#else
            gauss7(St,Kt);
#endif
            real_t G[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)G[i*D+j]=Kt[j*D+i];
            real_t Hx[D];m1(H,sp,Hx);real_t innov[D];for(int i=0;i<D;i++)innov[i]=z[s*dim+i]-Hx[i];
            real_t corr[D];m1(G,innov,corr);for(int i=0;i<D;i++)st[i]=sp[i]+corr[i];
            real_t KH[DS];m7(G,H,KH);for(int i=0;i<D;i++)for(int j=0;j<D;j++)KH[i*D+j]=(i==j?R_CONST(1.0):R_CONST(0.0))-KH[i*D+j];m7(KH,Pp,P);
        }
        for(int i=0;i<dim;i++)ss[b*dim+i]=st[i];for(int i=0;i<DS;i++)cs_mat[b*DS+i]=P[i];
    }
    printf("  GPU time: %.4f s\n",cpu_time()-t0);free(ss);free(cs_mat);free(ms);}
    return 0;
}
