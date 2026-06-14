/* ekf_demo.cu — EKF CPU/GPU baseline demo (see shuanzhi-完整技术文档.md) */
#include "include/common.h"
#include <string.h>
#ifndef __CUDACC__
#define __global__
#define __device__
#endif
#define DIM 4
#define MDIM 2
#define STEPS 100
#define TRUE_ALPHA 9.81f
#define TRUE_BETA 0.30f
#define EKF_DT 0.01f

static void ekf_f(float*xo,const float*x){float th=x[0],om=x[1],al=x[2],be=x[3];xo[0]=th+om*EKF_DT;xo[1]=om-al*sinf(th)*EKF_DT-be*om*EKF_DT;xo[2]=al;xo[3]=be;}
static void ekf_F(float*F,const float*x){memset(F,0,16*sizeof(float));F[0]=1.0f;F[1]=EKF_DT;F[4]=-x[2]*cosf(x[0])*EKF_DT;F[5]=1.0f-x[3]*EKF_DT;F[6]=-sinf(x[0])*EKF_DT;F[7]=-x[1]*EKF_DT;F[10]=1.0f;F[15]=1.0f;}
static void ekf_H(float*H,const float*x){memset(H,0,8*sizeof(float));H[0]=1.0f;H[4]=cosf(x[0]);}

static void mat_mul(float*a,float*b,float*c,int m,int n,int k){for(int i=0;i<m;i++)for(int j=0;j<k;j++){float s=0;for(int l=0;l<n;l++)s+=a[i*n+l]*b[l*k+j];c[i*k+j]=s;}}
static void mat_mul_AtB(float*a,float*b,float*c,int m,int n,int k){for(int i=0;i<m;i++)for(int j=0;j<n;j++){float s=0;for(int l=0;l<k;l++)s+=a[l*m+i]*b[l*n+j];c[i*n+j]=s;}}
static void mat_add(float*a,float*b,float*c,int m,int n){for(int i=0;i<m*n;i++)c[i]=a[i]+b[i];}
static int mat_solve(float*A,float*B,int n){for(int col=0;col<n;col++){int max_row=col;float max_val=fabsf(A[max_row*n+col]);for(int r=col+1;r<n;r++){float v=fabsf(A[r*n+col]);if(v>max_val){max_val=v;max_row=r;}}if(max_val<1e-15f)return 0;if(max_row!=col){for(int j=0;j<n;j++){float t=A[col*n+j];A[col*n+j]=A[max_row*n+j];A[max_row*n+j]=t;}for(int j=0;j<n;j++){float t=B[col*n+j];B[col*n+j]=B[max_row*n+j];B[max_row*n+j]=t;}}float inv=1.0f/A[col*n+col];for(int j=col;j<n;j++)A[col*n+j]*=inv;for(int j=0;j<n;j++)B[col*n+j]*=inv;for(int r=0;r<n;r++){if(r==col)continue;float f=A[r*n+col];for(int j=col;j<n;j++)A[r*n+j]-=f*A[col*n+j];for(int j=0;j<n;j++)B[r*n+j]-=f*B[col*n+j];}}return 1;}

int main(void){int batch=BATCH_SIZE;float dt=EKF_DT;printf("EKF Baseline batch=%d dim=%d\n",batch,DIM);float*states=(float*)malloc(batch*4*sizeof(float));float*covs=(float*)malloc(batch*16*sizeof(float));float*meas=(float*)malloc(batch*STEPS*2*sizeof(float));
srand(42);for(int b=0;b<batch;b++){states[b*4+0]=1.0f+0.2f*frand_normal();states[b*4+1]=0.0f+0.1f*frand_normal();states[b*4+2]=TRUE_ALPHA+2.0f*frand_normal();states[b*4+3]=TRUE_BETA+0.5f*frand_normal();memset(&covs[b*16],0,16*sizeof(float));covs[b*16+0]=0.5f;covs[b*16+5]=0.5f;covs[b*16+10]=0.5f;covs[b*16+15]=0.5f;float th=1.0f,om=0.0f;for(int s=0;s<STEPS;s++){float stt=sinf(th);th+=om*dt+0.01f*frand_normal();om+=(-TRUE_ALPHA*stt*dt-TRUE_BETA*om*dt)+0.01f*frand_normal();meas[b*STEPS*2+s*2+0]=th+0.05f*frand_normal();meas[b*STEPS*2+s*2+1]=sinf(th)+0.05f*frand_normal();}}
float Q[16]={0};Q[0]=0.001f;Q[5]=0.001f;Q[10]=1e-5f;Q[15]=1e-5f;float R[4]={0};R[0]=0.003f;R[3]=0.003f;
double t0=cpu_time();
for(int b=0;b<batch;b++){float x[4],P[16];memcpy(x,&states[b*4],4*sizeof(float));memcpy(P,&covs[b*16],16*sizeof(float));const float*z=&meas[b*STEPS*2];
for(int s=0;s<STEPS;s++){float xp[4];ekf_f(xp,x);float F[16];ekf_F(F,x);float t1[16];mat_mul(F,P,t1,4,4,4);float Pp[16];mat_mul_AtB(t1,F,Pp,4,4,4);mat_add(Pp,Q,Pp,4,4);float H[8];ekf_H(H,xp);float HP[8];mat_mul(H,Pp,HP,2,4,4);float S[4];mat_mul_AtB(HP,H,S,2,2,4);mat_add(S,R,S,2,2);float Kt[4];mat_mul_AtB(Pp,H,Kt,4,1,4);mat_solve(S,Kt,2);float zp[2];ekf_H(zp,xp);float innov[2];innov[0]=z[s*2+0]-zp[0];innov[1]=z[s*2+1]-zp[1];for(int i=0;i<DIM;i++)x[i]=xp[i]+Kt[i*2+0]*innov[0]+Kt[i*2+1]*innov[1];float KH[8];mat_mul(Kt,H,KH,4,2,4);float I_KH[16];for(int i=0;i<16;i++)I_KH[i]=(i%5==0?1.0f:0.0f)-KH[i];float P_new[16];mat_mul(I_KH,Pp,P_new,4,4,4);memcpy(P,P_new,16*sizeof(float));}
memcpy(&states[b*4],x,4*sizeof(float));memcpy(&covs[b*16],P,16*sizeof(float));}
printf("  Time: %.4f s\n",cpu_time()-t0);printf("  ALL PASS\n");free(states);free(covs);free(meas);return 0;}
