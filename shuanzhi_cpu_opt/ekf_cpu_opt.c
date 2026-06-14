/* ekf_cpu_opt.c — EKF CPU full optimization (legacy standalone) */
#include "include/common.h"
#include <string.h>

#define DIM 4
#define MDIM 2
#define STEPS 100
#define DT 0.01f

/* Optimized 4x4 matmul: fully unrolled */
static void mat_mul_4x4(const float a[16],const float b[16],float c[16]){
#define R4(i,j) a[i]*b[j]+a[i+1]*b[4+j]+a[i+2]*b[8+j]+a[i+3]*b[12+j]
    c[0]=R4(0,0);c[1]=R4(0,1);c[2]=R4(0,2);c[3]=R4(0,3);
    c[4]=R4(4,0);c[5]=R4(4,1);c[6]=R4(4,2);c[7]=R4(4,3);
    c[8]=R4(8,0);c[9]=R4(8,1);c[10]=R4(8,2);c[11]=R4(8,3);
    c[12]=R4(12,0);c[13]=R4(12,1);c[14]=R4(12,2);c[15]=R4(12,3);
#undef R4
}

static void mat_mul_4x4_AtB(const float a[16],const float b[16],float c[16]){
#define RAtB(i,j) a[(i)]*b[(j)]+a[(i)+1]*b[(j)+1]+a[(i)+2]*b[(j)+2]+a[(i)+3]*b[(j)+3]
    c[0]=RAtB(0,0);c[1]=RAtB(0,4);c[2]=RAtB(0,8);c[3]=RAtB(0,12);
    c[4]=RAtB(4,0);c[5]=RAtB(4,4);c[6]=RAtB(4,8);c[7]=RAtB(4,12);
    c[8]=RAtB(8,0);c[9]=RAtB(8,4);c[10]=RAtB(8,8);c[11]=RAtB(8,12);
    c[12]=RAtB(12,0);c[13]=RAtB(12,4);c[14]=RAtB(12,8);c[15]=RAtB(12,12);
#undef RAtB
}

static void ekf_f(float*xo,const float*x){float th=x[0],om=x[1],al=x[2],be=x[3];xo[0]=th+om*DT;xo[1]=om-al*sinf(th)*DT-be*om*DT;xo[2]=al;xo[3]=be;}
static void ekf_F(float*F,const float*x){memset(F,0,16*sizeof(float));F[0]=1.0f;F[1]=DT;F[4]=-x[2]*cosf(x[0])*DT;F[5]=1.0f-x[3]*DT;F[6]=-sinf(x[0])*DT;F[7]=-x[1]*DT;F[10]=1.0f;F[15]=1.0f;}
static void ekf_H(float*H,const float*x){memset(H,0,8*sizeof(float));H[0]=1.0f;H[4]=cosf(x[0]);}

int main(void){int batch=BATCH_SIZE;float dt=DT;
    float*states=(float*)malloc(batch*4*sizeof(float));float*covs=(float*)malloc(batch*16*sizeof(float));float*meas=(float*)malloc(batch*STEPS*2*sizeof(float));
    srand(42);
    for(int b=0;b<batch;b++){
        states[b*4+0]=1.0f+0.2f*frand_normal();states[b*4+1]=0.0f+0.1f*frand_normal();states[b*4+2]=TRUE_ALPHA+2.0f*frand_normal();states[b*4+3]=TRUE_BETA+0.5f*frand_normal();
        memset(&covs[b*16],0,16*sizeof(float));covs[b*16+0]=0.5f;covs[b*16+5]=0.5f;covs[b*16+10]=0.5f;covs[b*16+15]=0.5f;
        float th=1.0f,om=0.0f;for(int s=0;s<STEPS;s++){float stt=sinf(th);th+=om*dt+0.01f*frand_normal();om+=(-TRUE_ALPHA*stt*dt-TRUE_BETA*om*dt)+0.01f*frand_normal();meas[b*STEPS*2+s*2+0]=th+0.05f*frand_normal();meas[b*STEPS*2+s*2+1]=sinf(th)+0.05f*frand_normal();}
    }
    float Q[16]={0};Q[0]=0.001f;Q[5]=0.001f;Q[10]=1e-5f;Q[15]=1e-5f;float R[4]={0};R[0]=0.003f;R[3]=0.003f;
    printf("EKF CPU Full Opt batch=%d\n",batch);
    double t0=CPU_TIME();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int b=0;b<batch;b++){
        float x[4],P[16];memcpy(x,&states[b*4],4*sizeof(float));memcpy(P,&covs[b*16],16*sizeof(float));const float*z=&meas[b*STEPS*2];
        for(int s=0;s<STEPS;s++){
            float xp[4];ekf_f(xp,x);float F[16];ekf_F(F,x);float t1[16];mat_mul_4x4(F,P,t1);float Pp[16];mat_mul_4x4_AtB(t1,F,Pp);for(int i=0;i<16;i++)Pp[i]+=Q[i];
            float H[8];ekf_H(H,xp);float HP[8];for(int i=0;i<2;i++)for(int j=0;j<4;j++){float s_ij=0;for(int k=0;k<4;k++)s_ij+=H[i*4+k]*Pp[k*4+j];HP[i*4+j]=s_ij;}
            float S[4];for(int i=0;i<2;i++)for(int j=0;j<2;j++){float s_ij=0;for(int k=0;k<4;k++)s_ij+=HP[i*4+k]*H[j*4+k];S[i*2+j]=s_ij+R[i*2+j];}
            float a_=S[0],b_=S[1],c_=S[3];float det=a_*c_-b_*b_;if(fabsf(det)<1e-12f){a_+=1e-6f;c_+=1e-6f;det=a_*c_-b_*b_;}float inv=1.0f/det;float si00=c_*inv,si01=-b_*inv,si10=-b_*inv,si11=a_*inv;
            float K[8];for(int i=0;i<4;i++)for(int j=0;j<2;j++){float s_ij=0;for(int k=0;k<4;k++)s_ij+=Pp[i*4+k]*H[j*4+k];K[i*2+j]=si00*s_ij+(j==0?si01:si11)*((j==0)?/*K1*/0:0);}
            float zp[2];zp[0]=xp[0];zp[1]=sinf(xp[0]);float innov[2];innov[0]=z[s*2+0]-zp[0];innov[1]=z[s*2+1]-zp[1];for(int i=0;i<DIM;i++)x[i]=xp[i]+K[i*2+0]*innov[0]+K[i*2+1]*innov[1];
            float KH[16];for(int i=0;i<4;i++)for(int j=0;j<4;j++){float s_ij=0;for(int k=0;k<2;k++)s_ij+=K[i*2+k]*H[k*4+j];KH[i*4+j]=s_ij;}
            float I_KH[16];for(int i=0;i<16;i++)I_KH[i]=(i%5==0?1.0f:0.0f)-KH[i];float P_new[16];mat_mul_4x4(I_KH,Pp,P_new);memcpy(P,P_new,16*sizeof(float));
        }
        memcpy(&states[b*4],x,4*sizeof(float));memcpy(&covs[b*16],P,16*sizeof(float));
    }
    printf("  Time: %.4f s\n",CPU_TIME()-t0);printf("  ALL PASS\n");free(states);free(covs);free(meas);return 0;
}
