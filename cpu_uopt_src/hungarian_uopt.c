/* hungarian_uopt.c — Hungarian CPU unified optimization (8-level) */
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
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 8
#endif

#ifdef OPT_FLOAT
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

static inline double cpu_time(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return(double)ts.tv_sec+(double)ts.tv_nsec*1e-9;}
static inline real_t rand_real(void){return(real_t)rand()/(real_t)RAND_MAX;}
static inline real_t rand_range(real_t lo,real_t hi){return lo+(hi-lo)*rand_real();}

#ifdef OPT_ALGO
static void munkres(real_t*cost,int n,int*assignment){
    real_t cc[64];unsigned long long star_mask=0,prime_mask=0;unsigned row_covered=0,col_covered=0,star_cols=0;
    for(int i=0;i<n*n;i++)cc[i]=cost[i];
    for(int r=0;r<n;r++){real_t mv=cc[r*n];for(int c=1;c<n;c++)if(cc[r*n+c]<mv)mv=cc[r*n+c];for(int c=0;c<n;c++)cc[r*n+c]-=mv;}
    for(int c=0;c<n;c++){real_t mv=cc[c];for(int r=1;r<n;r++)if(cc[r*n+c]<mv)mv=cc[r*n+c];if(mv>R_CONST(1e-12))for(int r=0;r<n;r++)cc[r*n+c]-=mv;}
    unsigned rows_used=0,cols_used=0;
    for(int r=0;r<n;r++){if(rows_used&(1<<r))continue;for(int c=0;c<n;c++){if(cols_used&(1<<c))continue;if(cc[r*n+c]<=R_CONST(1e-12)){star_mask|=(1ULL<<(r*8+c));star_cols|=(1<<c);rows_used|=(1<<r);cols_used|=(1<<c);break;}}}
    for(;;){
        col_covered=0;for(int c=0;c<n;c++)if(star_cols&(1<<c))col_covered|=(1<<c);
        if(__builtin_popcount(col_covered)>=n)break;
        for(;;){
            int zero_r=-1,zero_c=-1;
            unsigned free_rows=(~row_covered)&((1u<<n)-1),free_cols=(~col_covered)&((1u<<n)-1);
            while(free_rows){int r=__builtin_ctz(free_rows);free_rows&=free_rows-1;unsigned cols=free_cols;while(cols){int c=__builtin_ctz(cols);cols&=cols-1;if(cc[r*n+c]<=R_CONST(1e-12)){zero_r=r;zero_c=c;goto found;}}}found:
            if(zero_r>=0){prime_mask|=(1ULL<<(zero_r*8+zero_c));unsigned rb=(unsigned)(star_mask>>(zero_r*8))&0xFF;int sc=rb?__builtin_ctz(rb):-1;
                if(sc>=0){row_covered|=(1<<zero_r);col_covered&=~(1<<sc);}
                else{int pr[24],pc[24],pl=1;pr[0]=zero_r;pc[0]=zero_c;
                    for(;;){int lc=pc[pl-1];unsigned long long cs=star_mask&(0x0101010101010101ULL<<lc);int sr=cs?__builtin_ctzll(cs)/8:-1;if(sr<0)break;pr[pl]=sr;pc[pl]=lc;pl++;unsigned pb=(unsigned)(prime_mask>>(sr*8))&0xFF;int pc2=pb?__builtin_ctz(pb):-1;pr[pl]=sr;pc[pl]=pc2;pl++;}
                    for(int p=0;p<pl;p++)star_mask^=(1ULL<<(pr[p]*8+pc[p]));
                    star_cols=0;unsigned long long tmp=star_mask;while(tmp){int bit=__builtin_ctzll(tmp);star_cols|=(1<<(bit&7));tmp&=tmp-1;}
                    prime_mask=0;row_covered=0;col_covered=0;break;}
            }else{real_t mv=R_INF;for(int r=0;r<n;r++){if(row_covered&(1<<r))continue;for(int c=0;c<n;c++){if(col_covered&(1<<c))continue;if(cc[r*n+c]<mv)mv=cc[r*n+c];}}
                for(int r=0;r<n;r++){for(int c=0;c<n;c++){if(row_covered&(1<<r))cc[r*n+c]+=mv;if(!(col_covered&(1<<c)))cc[r*n+c]-=mv;}}}}
    }
    for(int r=0;r<n;r++){unsigned rb=(unsigned)(star_mask>>(r*8))&0xFF;assignment[r]=rb?__builtin_ctz(rb):0;}
}
#else
static void munkres(real_t*cost,int n,int*assignment){
    signed char marks[64];char row_covered[8],col_covered[8];int star_cols[8];real_t cc[64];
    for(int i=0;i<n*n;i++)cc[i]=cost[i];
    memset(marks,0,n*n);memset(star_cols,0,n*sizeof(int));
    for(int r=0;r<n;r++){real_t mv=cc[r*n];for(int c=1;c<n;c++)if(cc[r*n+c]<mv)mv=cc[r*n+c];for(int c=0;c<n;c++)cc[r*n+c]-=mv;}
    for(int c=0;c<n;c++){real_t mv=cc[c];for(int r=1;r<n;r++)if(cc[r*n+c]<mv)mv=cc[r*n+c];for(int r=0;r<n;r++)cc[r*n+c]-=mv;}
    for(int r=0;r<n;r++){for(int c=0;c<n;c++){if(cc[r*n+c]<=R_CONST(1e-12)&&!star_cols[c]){marks[r*n+c]=1;star_cols[c]=1;break;}}}
    for(;;){
        memset(row_covered,0,n);memset(col_covered,0,n);int ccnt=0;for(int c=0;c<n;c++){if(star_cols[c]){col_covered[c]=1;ccnt++;}}if(ccnt>=n)break;
        for(;;){int zr=-1,zc=-1;
            for(int r=0;r<n&&zr<0;r++){if(row_covered[r])continue;for(int c=0;c<n;c++){if(col_covered[c])continue;if(cc[r*n+c]<=R_CONST(1e-12)){zr=r;zc=c;marks[zr*n+zc]=2;goto pd;}}}pd:
            if(zr<0){real_t mv=R_INF;for(int r=0;r<n;r++){if(row_covered[r])continue;for(int c=0;c<n;c++){if(col_covered[c])continue;if(cc[r*n+c]<mv)mv=cc[r*n+c];}}for(int r=0;r<n;r++){for(int c=0;c<n;c++){if(row_covered[r])cc[r*n+c]+=mv;if(!col_covered[c])cc[r*n+c]-=mv;}}zr=-1;zc=-1;continue;}
            int sc=-1;for(int c=0;c<n;c++)if(marks[zr*n+c]==1){sc=c;break;}
            if(sc<0){int pr[24],pc[24],pl=1;pr[0]=zr;pc[0]=zc;
                for(;;){int lc=pc[pl-1];int sr=-1;for(int r=0;r<n;r++)if(marks[r*n+lc]==1){sr=r;break;}if(sr<0)break;pr[pl]=sr;pc[pl]=lc;pl++;int pcol=-1;for(int c=0;c<n;c++)if(marks[sr*n+c]==2){pcol=c;break;}pr[pl]=sr;pc[pl]=pcol;pl++;}
                for(int p=0;p<pl;p++)marks[pr[p]*n+pc[p]]=(marks[pr[p]*n+pc[p]]==1)?0:1;
                memset(star_cols,0,n*sizeof(int));for(int r=0;r<n;r++)for(int c=0;c<n;c++)if(marks[r*n+c]==1)star_cols[c]=1;
                for(int r=0;r<n;r++)for(int c=0;c<n;c++)if(marks[r*n+c]==2)marks[r*n+c]=0;break;}
            row_covered[zr]=1;col_covered[sc]=0;zr=-1;zc=-1;
        }
    }
    for(int r=0;r<n;r++){int found=0;for(int c=0;c<n;c++){if(marks[r*n+c]==1){assignment[r]=c;found=1;break;}}if(!found)assignment[r]=0;}
}
#endif

int main(void){int n=MATRIX_SIZE,batch=BATCH_SIZE;
    printf("Hungarian CPU n=%d batch=%d",n,batch);
#ifdef OPT_OPENMP
    printf(" +OMP");
#endif
#ifdef OPT_SIMD
    printf(" +SIMD(N/A)");
#endif
#ifdef OPT_FLOAT
    printf(" +Float");
#endif
#ifdef OPT_ALGO
    printf(" +Algo(Bitmask)");
#endif
    printf("\n");
    printf("--- Test1: 4x4 OR-Tools ---\n");
    {real_t init[16]={90,75,75,80,35,85,55,65,125,95,90,105,45,110,95,115};real_t cost[16];int assign[4];memcpy(cost,init,16*sizeof(real_t));munkres(cost,4,assign);real_t ac=0;for(int i=0;i<4;i++)ac+=init[i*4+assign[i]];int exp[]={3,2,1,0};int pass=1;for(int i=0;i<4;i++)if(assign[i]!=exp[i])pass=0;printf("  Cost: %.0f %s\n",(double)ac,pass?"PASS":"FAIL");}
    printf("--- Test2: Brute force n=3-6 ---\n");
    {srand(12345);
    for(int tn=3;tn<=6;tn++){real_t cost[36],orig[36];int assign[6];for(int i=0;i<tn*tn;i++)orig[i]=cost[i]=rand_range(1,100);munkres(cost,tn,assign);
    int perm[12];for(int i=0;i<tn;i++)perm[i]=i;real_t best=R_INF;
    do{real_t total=0;for(int i=0;i<tn;i++)total+=orig[i*tn+perm[i]];if(total<best)best=total;}
    while([&](int*f,int*l){if(f==l)return 0;int*i=l-1;while(i>f){int*j=i--;if(*i<*j){int*k=l;while(*i>=*--k);int t=*i;*i=*k;*k=t;for(int*m=j,*r=l-1;m<r;m++,r--){t=*m;*m=*r;*r=t;}return 1;}}return 0;}(perm,perm+tn));
    real_t gc=0;for(int i=0;i<tn;i++)gc+=orig[i*tn+assign[i]];printf("  n=%d: brute=%.2f cpu=%.2f %s\n",tn,(double)best,(double)gc,R_ABS(gc-best)<R_CONST(1e-4)?"PASS":"FAIL");}}
    printf("--- Test3: Batch perf ---\n");
    {real_t*costs=(real_t*)malloc((size_t)batch*n*n*sizeof(real_t));int*assigns=(int*)malloc((size_t)batch*n*sizeof(int));
    srand(42);for(int b=0;b<batch;b++)for(int i=0;i<n;i++)for(int j=0;j<n;j++)costs[b*n*n+i*n+j]=rand_range(1,100);
    double t0=cpu_time();
#ifdef OPT_OPENMP
#pragma omp parallel for
#endif
    for(int b=0;b<batch;b++)munkres(&costs[b*n*n],n,&assigns[b*n]);
    printf("  CPU time: %.4f s\n",cpu_time()-t0);free(costs);free(assigns);}
    return 0;
}
