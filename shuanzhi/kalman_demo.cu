/* kalman_demo.cu — Kalman CPU/GPU baseline demo (see shuanzhi-完整技术文档.md) */
#include "include/common.h"
#include <string.h>
#ifndef __CUDACC__
#define __global__
#define __device__
#endif
#define DIM 7
#define DSQ 49
#define STEPS 100

static void mat_mul(float*a,float*b,float*c,int m,int n,int k){for(int i=0;i<m;i++)for(int j=0;j<k;j++){float s=0;for(int l=0;l<n;l++)s+=a[i*n+l]*b[l*k+j];c[i*k+j]=s;}}
static void mat_mul_AtB(float*a,float*b,float*c,int m,int n,int k){for(int i=0;i<m;i++)for(int j=0;j<n;j++){float s=0;for(int l=0;l<k;l++)s+=a[l*m+i]*b[l*n+j];c[i*n+j]=s;}}
static void mat_add(float*a,float*b,float*c,int m,int n){for(int i=0;i<m*n;i++)c[i]=a[i]+b[i];}
static int mat_solve(float*A,float*B,int n){for(int col=0;col<n;col++){int mr=col;float mv=fabsf(A[mr*n+col]);for(int r=col+1;r<n;r++){float v=fabsf(A[r*n+col]);if(v>mv){mv=v;mr=r;}}if(mv<1e-15f)return 0;if(mr!=col){for(int j=0;j<n;j++){float t=A[col*n+j];A[col*n+j]=A[mr*n+j];A[mr*n+j]=t;}for(int j=0;j<n;j++){float t=B[col*n+j];B[col*n+j]=B[mr*n+j];B[mr*n+j]=t;}}float inv=1.0f/A[col*n+col];for(int j=col;j<n;j++)A[col*n+j]*=inv;for(int j=0;j<n;j++)B[col*n+j]*=inv;for(int r=0;r<n;r++){if(r==col)continue;float f=A[r*n+col];for(int j=col;j<n;j++)A[r*n+j]-=f*A[col*n+j];for(int j=0;j<n;j++)B[r*n+j]-=f*B[col*n+j];}}return 1;}
static void kalman_predict(float*x,float*P,float*F,float*Q,int dim){float xp[7];mat_mul(F,x,xp,dim,dim,1);memcpy(x,xp,dim*sizeof(float));float FP[49];mat_mul(F,P,FP,dim,dim,dim);float Pp[49];mat_mul_AtB(FP,F,Pp,dim,dim,dim);mat_add(Pp,Q,P,dim,dim);}
static void kalman_update(float*x,float*P,float*H,float*R,float*z,int dim){float HP[49];mat_mul(H,P,HP,dim,dim,dim);float S[49];mat_mul_AtB(HP,H,S,dim,dim,dim);mat_add(S,R,S,dim,dim);float Hx[7];mat_mul(H,x,Hx,dim,dim,1);float y[7];for(int d=0;d<dim;d++)y[d]=z[d]-Hx[d];float PHt[49];mat_mul_AtB(P,H,PHt,dim,dim,dim);float Kt[49];memcpy(Kt,PHt,sizeof(PHt));mat_solve(S,Kt,dim);for(int d=0;d<dim;d++)for(int k=0;k<dim;k++)x[d]+=Kt[k*dim+d]*y[k];float KH[49];mat_mul(Kt,H,KH,dim,dim,dim);float KHP[49];mat_mul(KH,P,KHP,dim,dim,dim);for(int i=0;i<dim*dim;i++)P[i]-=KHP[i];}

int main(void){int dim=DIM,steps=STEPS,batch=BATCH_SIZE;printf("Kalman Baseline dim=%d batch=%d\n",dim,batch);float F[49]={0},H[49]={0},Q[49]={0},R[49]={0},Rn[49]={0};for(int i=0;i<dim;i++){F[i*dim+i]=1.0f;H[i*dim+i]=1.0f;Q[i*dim+i]=1.0f;Rn[i*dim+i]=1.0f;}float sigma=1.0f;for(int i=0;i<dim*dim;i++)Rn[i]*=sigma*sigma;
float*ss=(float*)malloc(batch*dim*sizeof(float));float*cs=(float*)malloc(batch*dim*dim*sizeof(float));float*ms=(float*)malloc(batch*steps*dim*sizeof(float));
srand(42);for(int b=0;b<batch;b++){for(int i=0;i<dim;i++)ss[b*dim+i]=0.0f;for(int i=0;i<dim;i++)for(int j=0;j<dim;j++)cs[b*dim*dim+i*dim+j]=(i==j)?1.0f:0.0f;for(int s=0;s<steps;s++){float st[7];for(int i=0;i<dim;i++)st[i]=frand_range(-1.0f,1.0f);for(int i=0;i<dim;i++){float m=st[i];for(int j=0;j<dim;j++)m+=frand_range(-sigma,sigma)/*noise*/;ms[b*steps*dim+s*dim+i]=m;}}}
printf("--- Test1: No noise ---\n");
{float st[7],se[7],P[49],meas[700];srand(42);for(int i=0;i<dim;i++)st[i]=frand_range(-1.0f,1.0f);memcpy(meas,st,dim*sizeof(float));for(int s=1;s<steps;s++){float xt[7];mat_mul(F,st,xt,dim,dim,1);memcpy(st,xt,dim*sizeof(float));memcpy(&meas[s*dim],st,dim*sizeof(float));}memset(se,0,dim*sizeof(float));for(int i=0;i<dim;i++)for(int j=0;j<dim;j++)P[i*dim+j]=(i==j)?1.0f:0.0f;
double t0=cpu_time();
for(int s=0;s<steps;s++){kalman_predict(se,P,F,Q,dim);kalman_update(se,P,H,R,&meas[s*dim],dim);}
printf("  Time: %.4f s\n",cpu_time()-t0);float me=0;for(int i=0;i<dim;i++){float e=fabsf(se[i]-st[i]);if(e>me)me=e;}printf("  Max err: %e %s\n",me,me<1e-4f?"PASS":"FAIL");}
free(ss);free(cs);free(ms);return 0;}
