/* hungarian_demo.cu — Hungarian CPU/GPU baseline demo (see shuanzhi-完整技术文档.md) */
#include "include/common.h"
#include <string.h>
#ifndef __CUDACC__
#define __global__
#define __device__
#endif
#define N 8

static void munkres(float*cost,int n,int*assignment){
    int i,j,r,c;float row_min,col_min;
    for(r=0;r<n;r++){row_min=cost[r*n];for(c=1;c<n;c++)if(cost[r*n+c]<row_min)row_min=cost[r*n+c];for(c=0;c<n;c++)cost[r*n+c]-=row_min;}
    for(c=0;c<n;c++){col_min=cost[c];for(r=1;r<n;r++)if(cost[r*n+c]<col_min)col_min=cost[r*n+c];for(r=0;r<n;r++)cost[r*n+c]-=col_min;}
    int*sc=(int*)calloc(n,sizeof(int));int*rc=(int*)calloc(n,sizeof(int));int*cc=(int*)calloc(n,sizeof(int));
    memset(assignment,-1,n*sizeof(int));
    for(r=0;r<n;r++){for(c=0;c<n;c++){if(cost[r*n+c]<=1e-12f&&!sc[c]){assignment[r]=c;sc[c]=1;break;}}}
    for(;;){
        memset(rc,0,n*sizeof(int));memset(cc,0,n*sizeof(int));int covered=0;
        for(r=0;r<n;r++)if(assignment[r]<0)rc[r]=1;
        int changed;do{changed=0;
            for(r=0;r<n;r++)if(rc[r])for(c=0;c<n;c++)if(cost[r*n+c]<=1e-12f&&!cc[c]){cc[c]=1;changed=1;}
            for(c=0;c<n;c++)if(cc[c])for(r=0;r<n;r++)if(assignment[r]==c&&!rc[r]){rc[r]=1;changed=1;}
        }while(changed);
        for(r=0;r<n;r++)covered+=rc[r]+cc[r];
        if(covered==n)break;
        float mv=1e30f;for(r=0;r<n;r++)if(!rc[r])for(c=0;c<n;c++)if(!cc[c]&&cost[r*n+c]<mv)mv=cost[r*n+c];
        for(r=0;r<n;r++)for(c=0;c<n;c++){if(!rc[r]&&!cc[c])cost[r*n+c]-=mv;else if(rc[r]&&cc[c])cost[r*n+c]+=mv;}
    }
    free(sc);free(rc);free(cc);
}

int main(void){int n=N,batch=BATCH_SIZE;printf("Hungarian Baseline n=%d batch=%d\n",n,batch);
printf("--- Test1: 4x4 OR-Tools ---\n");
{float init[16]={90,75,75,80,35,85,55,65,125,95,90,105,45,110,95,115};float cost[16];int assign[4];memcpy(cost,init,16*sizeof(float));munkres(cost,4,assign);float ac=0;for(int i=0;i<4;i++)ac+=init[i*4+assign[i]];int exp[]={3,2,1,0};int pass=1;for(int i=0;i<4;i++)if(assign[i]!=exp[i])pass=0;printf("  Cost: %.0f %s\n",ac,pass?"PASS":"FAIL");}
printf("--- Test2: Batch perf ---\n");
{float*costs=(float*)malloc(batch*n*n*sizeof(float));int*assigns=(int*)malloc(batch*n*sizeof(int));srand(42);for(int b=0;b<batch;b++)for(int i=0;i<n*n;i++)costs[b*n*n+i]=frand_range(1.0f,100.0f);
double t0=cpu_time();for(int b=0;b<batch;b++)munkres(&costs[b*n*n],n,&assigns[b*n]);printf("  CPU time: %.4f s\n",cpu_time()-t0);free(costs);free(assigns);}
return 0;}
