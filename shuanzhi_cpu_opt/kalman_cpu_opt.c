/* kalman_cpu_opt.c — Kalman CPU full optimization (legacy standalone) */
#include "include/common.h"
#include <string.h>
#define D 7
#define DS 49

static void m7(const float a[DS],const float b[DS],float c[DS]){
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
static void mbt(const float a[DS],const float b[DS],float c[DS]){for(int i=0;i<D;i++){const float*ar=&a[i*D];for(int j=0;j<D;j++){const float*br=&b[j*D];c[i*D+j]=ar[0]*br[0]+ar[1]*br[1]+ar[2]*br[2]+ar[3]*br[3]+ar[4]*br[4]+ar[5]*br[5]+ar[6]*br[6];}}}
static int chol7(float A[DS]){for(int j=0;j<D;j++){float s=0;for(int k=0;k<j;k++){float v=A[j*D+k];s+=v*v;}float d=A[j*D+j]-s;if(d<=1e-12f)return 0;A[j*D+j]=sqrtf(d);for(int i=j+1;i<D;i++){s=0;for(int k=0;k<j;k++)s+=A[i*D+k]*A[j*D+k];A[i*D+j]=(A[i*D+j]-s)/A[j*D+j];}}return 1;}
static void cs7(const float L[DS],float B[DS]){for(int c=0;c<D;c++){for(int i=0;i<D;i++){float s=B[i*D+c];for(int k=0;k<i;k++)s-=L[i*D+k]*B[k*D+c];B[i*D+c]=s/L[i*D+i];}}for(int c=0;c<D;c++){for(int i=D-1;i>=0;i--){float s=B[i*D+c];for(int k=i+1;k<D;k++)s-=L[k*D+i]*B[k*D+c];B[i*D+c]=s/L[i*D+i];}}}
static void mid7(float m[DS]){for(int i=0;i<DS;i++)m[i]=0.0f;for(int i=0;i<D;i++)m[i*D+i]=1.0f;}

int main(void){int dim=D,steps=100,batch=BATCH_SIZE;printf("Kalman CPU Full Opt dim=%d batch=%d\n",dim,batch);
    float F[DS],H[DS],Q[DS],R[DS];mid7(F);mid7(H);mid7(Q);for(int i=0;i<DS;i++)R[i]=0.0f;
    float*ss=(float*)malloc(batch*dim*sizeof(float));float*cs_mat=(float*)malloc(batch*DS*sizeof(float));float*ms=(float*)malloc(batch*steps*dim*sizeof(float));
    srand(42);for(int b=0;b<batch;b++){for(int i=0;i<dim;i++)ss[b*dim+i]=0.0f;mid7(&cs_mat[b*DS]);for(int s=0;s<steps;s++)for(int i=0;i<dim;i++)ms[b*steps*dim+s*dim+i]=frand_range(-1.0f,1.0f)+frand_range(-1.0f,1.0f);}
    double t0=CPU_TIME();
#pragma omp parallel for
    for(int b=0;b<batch;b++){
        float st[D],P[DS];for(int i=0;i<dim;i++)st[i]=ss[b*dim+i];for(int i=0;i<DS;i++)P[i]=cs_mat[b*DS+i];const float*z=&ms[b*steps*dim];
        for(int s=0;s<steps;s++){
            float sp[D];for(int i=0;i<D;i++){float sum=0;for(int k=0;k<D;k++)sum+=F[i*D+k]*st[k];sp[i]=sum;}
            float tFP[DS];m7(F,P,tFP);float Pp[DS];mbt(tFP,F,Pp);for(int i=0;i<DS;i++)Pp[i]+=Q[i];
            float HP[DS],St[DS];m7(H,Pp,HP);mbt(HP,H,St);for(int i=0;i<DS;i++)St[i]+=R[i];
            float Kt[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)Kt[i*D+j]=HP[j*D+i];
            chol7(St);cs7(St,Kt);
            float G[DS];for(int i=0;i<D;i++)for(int j=0;j<D;j++)G[i*D+j]=Kt[j*D+i];
            float Hx[D];for(int i=0;i<D;i++){float sum=0;for(int k=0;k<D;k++)sum+=H[i*D+k]*sp[k];Hx[i]=sum;}
            float innov[D];for(int i=0;i<D;i++)innov[i]=z[s*dim+i]-Hx[i];
            for(int i=0;i<D;i++){float sum=0;for(int k=0;k<D;k++)sum+=G[i*D+k]*innov[k];st[i]=sp[i]+sum;}
            float KH[DS];m7(G,H,KH);for(int i=0;i<D;i++)for(int j=0;j<D;j++)KH[i*D+j]=(i==j?1.0f:0.0f)-KH[i*D+j];m7(KH,Pp,P);
        }
        for(int i=0;i<dim;i++)ss[b*dim+i]=st[i];for(int i=0;i<DS;i++)cs_mat[b*DS+i]=P[i];
    }
    printf("  Time: %.4f s\n",CPU_TIME()-t0);free(ss);free(cs_mat);free(ms);return 0;
}
