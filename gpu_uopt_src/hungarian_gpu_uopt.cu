/* hungarian_gpu_uopt.cu — Hungarian GPU unified optimization */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 8
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 1024
#endif

#ifdef OPT_FP32
typedef float real_t;
#define R_ABS fabsf
#define R_INF 1e30f
#define R_CONST(v) v##f
#else
typedef double real_t;
#define R_ABS fabs
#define R_INF 1e30
#define R_CONST(v) v
#endif

static inline float gpu_elapsed(cudaEvent_t s,cudaEvent_t e){float ms;cudaEventElapsedTime(&ms,s,e);return ms/1000.0f;}
static inline real_t rand_real(void){return (real_t)rand()/(real_t)RAND_MAX;}
static inline real_t rand_range(real_t lo,real_t hi){return lo+(hi-lo)*rand_real();}

#ifdef OPT_ALGO
#define GPU_CTZ(x) (__ffs(x)-1)
#define GPU_CTZLL(x) (__ffsll(x)-1)
#define GPU_POPC(x) __popc(x)

__global__ void hungarian_kernel(const real_t *costs,int *results,int n,int batch){
    int pid=blockIdx.x*blockDim.x+threadIdx.x;
    if(pid>=batch)return;
    real_t cost[64];unsigned long long star_mask=0,prime_mask=0;unsigned row_covered=0,col_covered=0,star_cols=0;
    const real_t*src=&costs[pid*n*n];
    for(int i=0;i<n*n;i++)cost[i]=src[i];
    /* step1: row reduce */
    for(int r=0;r<n;r++){real_t mv=cost[r*n];for(int c=1;c<n;c++)if(cost[r*n+c]<mv)mv=cost[r*n+c];for(int c=0;c<n;c++)cost[r*n+c]-=mv;}
    /* step1b: col reduce */
    for(int c=0;c<n;c++){real_t mv=cost[c];for(int r=1;r<n;r++)if(cost[r*n+c]<mv)mv=cost[r*n+c];if(mv>R_CONST(1e-12))for(int r=0;r<n;r++)cost[r*n+c]-=mv;}
    /* step2: star zeroes */
    unsigned rows_used=0,cols_used=0;
    for(int r=0;r<n;r++){if(rows_used&(1<<r))continue;for(int c=0;c<n;c++){if(cols_used&(1<<c))continue;if(cost[r*n+c]<=R_CONST(1e-12)){star_mask|=(1ULL<<(r*8+c));star_cols|=(1<<c);rows_used|=(1<<r);cols_used|=(1<<c);break;}}}
    for(;;){
        col_covered=0;for(int c=0;c<n;c++)if(star_cols&(1<<c))col_covered|=(1<<c);
        if(GPU_POPC(col_covered)>=n)break;
        for(;;){
            int zero_r=-1,zero_c=-1;
            unsigned free_rows=(~row_covered)&((1u<<n)-1),free_cols=(~col_covered)&((1u<<n)-1);
            while(free_rows){int r=GPU_CTZ(free_rows);free_rows&=free_rows-1;unsigned cols=free_cols;while(cols){int c=GPU_CTZ(cols);cols&=cols-1;if(cost[r*n+c]<=R_CONST(1e-12)){zero_r=r;zero_c=c;goto found;} }} found:
            if(zero_r>=0){prime_mask|=(1ULL<<(zero_r*8+zero_c));unsigned rb=(unsigned)(star_mask>>(zero_r*8))&0xFF;int sc=rb?GPU_CTZ(rb):-1;
                if(sc>=0){row_covered|=(1<<zero_r);col_covered&=~(1<<sc);}
                else{int pr[24],pc[24],pl=1;pr[0]=zero_r;pc[0]=zero_c;
                    for(;;){int lc=pc[pl-1];unsigned long long cs=star_mask&(0x0101010101010101ULL<<lc);int sr=cs?GPU_CTZLL(cs)/8:-1;if(sr<0)break;pr[pl]=sr;pc[pl]=lc;pl++;unsigned pb=(unsigned)(prime_mask>>(sr*8))&0xFF;int pc2=pb?GPU_CTZ(pb):-1;pr[pl]=sr;pc[pl]=pc2;pl++;}
                    for(int p=0;p<pl;p++)star_mask^=(1ULL<<(pr[p]*8+pc[p]));
                    star_cols=0;unsigned long long tmp=star_mask;while(tmp){int bit=GPU_CTZLL(tmp);star_cols|=(1<<(bit&7));tmp&=tmp-1;}
                    prime_mask=0;row_covered=0;col_covered=0;break;}
            }else{real_t mv=R_INF;for(int r=0;r<n;r++){if(row_covered&(1<<r))continue;for(int c=0;c<n;c++){if(col_covered&(1<<c))continue;if(cost[r*n+c]<mv)mv=cost[r*n+c];}}
                for(int r=0;r<n;r++){for(int c=0;c<n;c++){if(row_covered&(1<<r))cost[r*n+c]+=mv;if(!(col_covered&(1<<c)))cost[r*n+c]-=mv;}}}
        }
    }
    for(int r=0;r<n;r++){unsigned rb=(unsigned)(star_mask>>(r*8))&0xFF;results[pid*n+r]=rb?GPU_CTZ(rb):0;}
}
#else
#define STAR 1
#define PRIME 2
__global__ void hungarian_kernel(const real_t *costs,int *results,int n,int batch){
    int pid=blockIdx.x*blockDim.x+threadIdx.x;
    if(pid>=batch)return;
    signed char marks[64];char row_covered[8],col_covered[8];int star_cols[8];real_t cost[64];
    const real_t*src=&costs[pid*n*n];for(int i=0;i<n*n;i++)cost[i]=src[i];
    memset(marks,0,n*n);memset(star_cols,0,n*sizeof(int));
    for(int r=0;r<n;r++){real_t mv=cost[r*n];for(int c=1;c<n;c++)if(cost[r*n+c]<mv)mv=cost[r*n+c];for(int c=0;c<n;c++)cost[r*n+c]-=mv;}
    for(int c=0;c<n;c++){real_t mv=cost[c];for(int r=1;r<n;r++)if(cost[r*n+c]<mv)mv=cost[r*n+c];for(int r=0;r<n;r++)cost[r*n+c]-=mv;}
    for(int r=0;r<n;r++){for(int c=0;c<n;c++){if(cost[r*n+c]<=R_CONST(1e-12)&&!star_cols[c]){marks[r*n+c]=STAR;star_cols[c]=1;break;}}}
    for(;;){
        memset(row_covered,0,n);memset(col_covered,0,n);int cc=0;for(int c=0;c<n;c++){if(star_cols[c]){col_covered[c]=1;cc++;}}if(cc>=n)break;
        for(;;){int zr=-1,zc=-1;
            for(int r=0;r<n&&zr<0;r++){if(row_covered[r])continue;for(int c=0;c<n;c++){if(col_covered[c])continue;if(cost[r*n+c]<=R_CONST(1e-12)){zr=r;zc=c;marks[zr*n+zc]=PRIME;goto pd;}}}
            pd:if(zr<0){real_t mv=R_INF;for(int r=0;r<n;r++){if(row_covered[r])continue;for(int c=0;c<n;c++){if(col_covered[c])continue;if(cost[r*n+c]<mv)mv=cost[r*n+c];}}for(int r=0;r<n;r++){for(int c=0;c<n;c++){if(row_covered[r])cost[r*n+c]+=mv;if(!col_covered[c])cost[r*n+c]-=mv;}}zr=-1;zc=-1;continue;}
            int sc=-1;for(int c=0;c<n;c++)if(marks[zr*n+c]==STAR){sc=c;break;}
            if(sc<0){int pr[24],pc[24],pl=1;pr[0]=zr;pc[0]=zc;
                for(;;){int lc=pc[pl-1];int sr=-1;for(int r=0;r<n;r++)if(marks[r*n+lc]==STAR){sr=r;break;}if(sr<0)break;pr[pl]=sr;pc[pl]=lc;pl++;int pcol=-1;for(int c=0;c<n;c++)if(marks[sr*n+c]==PRIME){pcol=c;break;}pr[pl]=sr;pc[pl]=pcol;pl++;}
                for(int p=0;p<pl;p++)marks[pr[p]*n+pc[p]]=(marks[pr[p]*n+pc[p]]==STAR)?0:STAR;
                memset(star_cols,0,n*sizeof(int));for(int r=0;r<n;r++)for(int c=0;c<n;c++)if(marks[r*n+c]==STAR)star_cols[c]=1;
                for(int r=0;r<n;r++)for(int c=0;c<n;c++)if(marks[r*n+c]==PRIME)marks[r*n+c]=0;break;}
            row_covered[zr]=1;col_covered[sc]=0;zr=-1;zc=-1;
        }
    }
    for(int r=0;r<n;r++){int found=0;for(int c=0;c<n;c++){if(marks[r*n+c]==STAR){results[pid*n+r]=c;found=1;break;}}if(!found)results[pid*n+r]=0;}
}
#endif

void munkres_gpu(const real_t *hc,int *hr,int n,int batch,double *gt){
    real_t *dc;int *dr;cudaMalloc(&dc,(size_t)batch*n*n*sizeof(real_t));cudaMalloc(&dr,(size_t)batch*n*sizeof(int));
    cudaMemcpy(dc,hc,(size_t)batch*n*n*sizeof(real_t),cudaMemcpyHostToDevice);
    int th=128,bl=(batch+th-1)/th;
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    cudaEventRecord(s);hungarian_kernel<<<bl,th>>>(dc,dr,n,batch);cudaEventRecord(e);cudaEventSynchronize(e);
    *gt=(double)gpu_elapsed(s,e);
    cudaMemcpy(hr,dr,(size_t)batch*n*sizeof(int),cudaMemcpyDeviceToHost);
    cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dc);cudaFree(dr);
}

int main(void){int n=MATRIX_SIZE,batch=BATCH_SIZE;
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("Hungarian GPU n=%d batch=%d GPU=%s",n,batch,p.name);
#ifdef OPT_FP32
    printf(" float");
#else
    printf(" double");
#endif
#ifdef OPT_ALGO
    printf("+Bitmask");
#endif
    printf("\n");
    /* Test1: OR-Tools 4x4 */
    printf("--- Test1: 4x4 OR-Tools ---\n");
    {real_t init[16]={90,75,75,80,35,85,55,65,125,95,90,105,45,110,95,115};real_t cost[16];int result[4];double gt;
    memcpy(cost,init,16*sizeof(real_t));munkres_gpu(cost,result,4,1,&gt);
    real_t ac=0;for(int i=0;i<4;i++)ac+=init[i*4+result[i]];int exp[]={3,2,1,0};int pass=1;for(int i=0;i<4;i++)if(result[i]!=exp[i])pass=0;
    printf("  Cost: %.0f %s\n",(double)ac,pass?"PASS":"FAIL");}
    /* Test2: brute force n=3..6 */
    printf("--- Test2: Brute force ---\n");
    {srand(12345);
    for(int tn=3;tn<=6;tn++){real_t cost[36],orig[36];int result[6];double gt;
    for(int i=0;i<tn*tn;i++)orig[i]=cost[i]=rand_range(1,100);
    munkres_gpu(cost,result,tn,1,&gt);
    /* brute force */ int perm[12];for(int i=0;i<tn;i++)perm[i]=i;real_t best=R_INF;
    do{real_t total=0;for(int i=0;i<tn;i++)total+=orig[i*tn+perm[i]];if(total<best)best=total;}
    while([&](int*f,int*l){if(f==l)return 0;int*i=l-1;while(i>f){int*j=i--;if(*i<*j){int*k=l;while(*i>=*--k);int t=*i;*i=*k;*k=t;for(int*m=j,*r=l-1;m<r;m++,r--){t=*m;*m=*r;*r=t;}return 1;}}return 0;}(perm,perm+tn));
    real_t gc=0;for(int i=0;i<tn;i++)gc+=orig[i*tn+result[i]];
    printf("  n=%d: brute=%.2f gpu=%.2f %s\n",tn,(double)best,(double)gc,R_ABS(gc-best)<R_CONST(1e-4)?"PASS":"FAIL");}}
    /* Test3: batch perf */
    printf("--- Test3: Batch perf ---\n");
    {real_t*hc=(real_t*)malloc((size_t)batch*n*n*sizeof(real_t));int*hr=(int*)malloc((size_t)batch*n*sizeof(int));double gt;
    srand(42);for(int b=0;b<batch;b++)for(int i=0;i<n;i++)for(int j=0;j<n;j++)hc[b*n*n+i*n+j]=rand_range(1,100);
    munkres_gpu(hc,hr,n,batch,&gt);printf("  GPU time: %.4f s\n",gt);free(hc);free(hr);}
    return 0;
}
