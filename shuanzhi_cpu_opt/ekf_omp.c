/* ekf_omp.c — EKF CPU OpenMP optimization (legacy standalone) */
#include "include/common.h"
#include <string.h>

#define DIM 4
#define MDIM 2
#define STEPS 100
#define DT 0.01f

static void mat_mul(float*a,float*b,float*c,int m,int n,int k){for(int i=0;i<m;i++)for(int j=0;j<k;j++){float s=0;for(int l=0;l<n;l++)s+=a[i*n+l]*b[l*k+j];c[i*k+j]=s;}}
static void mat_mul_AtB(float*a,float*b,float*c,int m,int n,int k){for(int i=0;i<m;i++)for(int j=0;j<n;j++){float s=0;for(int l=0;l<k;l++)s+=a[l*m+i]*b[l*n+j];c[i*n+j]=s;}}
static void ekf_f(float*xo,const float*x){float th=x[0],om=x[1],al=x[2],be=x[3];xo[0]=th+om*DT;xo[1]=om-al*sinf(th)*DT-be*om*DT;xo[2]=al;xo[3]=be;}
static void ekf_F(float*F,const float*x){memset(F,0,16*sizeof(float));F[0]=1.0f;F[1]=DT;F[4]=-x[2]*cosf(x[0])*DT;F[5]=1.0f-x[3]*DT;F[6]=-sinf(x[0])*DT;F[7]=-x[1]*DT;F[10]=1.0f;F[15]=1.0f;}

int main(void){int batch=BATCH_SIZE;float dt=DT;printf("EKF CPU OMP batch=%d\n",batch);
    float*states=(float*)malloc(batch*4*sizeof(float));float*covs=(float*)malloc(batch*16*sizeof(float));float*meas=(float*)malloc(batch*STEPS*2*sizeof(float));
    srand(42);for(int b=0;b<batch;b++){states[b*4+0]=1.0f+0.2f*frand_normal();states[b*4+1]=0.0f+0.1f*frand_normal();states[b*4+2]=TRUE_ALPHA+2.0f*frand_normal();states[b*4+3]=TRUE_BETA+0.5f*frand_normal();memset(&covs[b*16],0,16*sizeof(float));covs[b*16+0]=0.5f;covs[b*16+5]=0.5f;covs[b*16+10]=0.5f;covs[b*16+15]=0.5f;float th=1.0f,om=0.0f;for(int s=0;s<STEPS;s++){float stt=sinf(th);th+=om*dt+0.01f*frand_normal();om+=(-TRUE_ALPHA*stt*dt-TRUE_BETA*om*dt)+0.01f*frand_normal();meas[b*STEPS*2+s*2+0]=th+0.05f*frand_normal();meas[b*STEPS*2+s*2+1]=sinf(th)+0.05f*frand_normal();}}
    float Q[16]={0},R[4]={0};Q[0]=0.001f;Q[5]=0.001f;Q[10]=1e-5f;Q[15]=1e-5f;R[0]=0.003f;R[3]=0.003f;
    double t0=CPU_TIME();
#pragma omp parallel for
    for(int b=0;b<batch;b++){
        float x[4],P[16];memcpy(x,&states[b*4],4*sizeof(float));memcpy(P,&covs[b*16],16*sizeof(float));const float*z=&meas[b*STEPS*2];
        for(int s=0;s<STEPS;s++){
            float xp[4];ekf_f(xp,x);float F[16];ekf_F(F,x);float t1[16];mat_mul(F,P,t1,4,4,4);float Pp[16];mat_mul_AtB(t1,F,Pp,4,4,4);for(int i=0;i<16;i++)Pp[i]+=Q[i];
            float H[8]={0};H[0]=1.0f;H[4]=cosf(xp[0]);float HP[8];mat_mul(H,Pp,HP,2,4,4);float S[4];mat_mul_AtB(HP,H,S,2,2,4);for(int i=0;i<4;i++)S[i]+=R[i];
            float a_=S[0],b_=S[1],c_=S[3];float det=a_*c_-b_*b_;if(fabsf(det)<1e-12f){a_+=1e-6f;c_+=1e-6f;det=a_*c_-b_*b_;}float inv=1.0f/det;
            float K[8];for(int i=0;i<4;i++)for(int j=0;j<2;j++){float sum=0;for(int k=0;k<4;k++)sum+=Pp[i*4+k]*H[j*4+k];K[i*2+j]=sum*inv;}
            float zp[2];zp[0]=xp[0];zp[1]=sinf(xp[0]);float innov[2];innov[0]=z[s*2+0]-zp[0];innov[1]=z[s*2+1]-zp[1];for(int i=0;i<DIM;i++)x[i]=xp[i]+K[i*2+0]*innov[0]+K[i*2+1]*innov[1];
        }
    }
    printf("  Time: %.4f s\n",CPU_TIME()-t0);free(states);free(covs);free(meas);return 0;
}
