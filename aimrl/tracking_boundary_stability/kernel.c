/* Independent compact-encoder kernel. All learning and environment parameters
 * are supplied by Python. No fast math or contraction. Float32 Q increments. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
typedef double (*next_double)(void *);
typedef uint32_t (*next_uint32)(void *);
static const int dx[]={-1,1,0,0,0,0,-3,3,0,0},dy[]={0,0,-1,1,0,0,0,0,-3,3};
static const int hx[]={1,0,-1,0},hy[]={0,1,0,-1};
static int clip(int x){return x<0?0:x>24?24:x;}
static int row_of(const int32_t *map,int w,int xc,int yc,int xt,int yt,int h,int s){
    return (((map[xc*25+xt]*w+map[yc*25+yt])*4+h)*2+s-1)*10;
}
static int greedy(const float *q){int b=0;for(int a=1;a<10;a++)if(q[a]>q[b])b=a;return b;}
static int random_action(void *rng,next_uint32 u){
    uint64_t v;do{v=(uint64_t)u(rng)*10u;}while((uint32_t)v<6u);return (int)(v>>32);
}
API void rollout(float *q,uint32_t *counts,const int32_t *map,int w,
                 const int32_t *initial,const double *draws,int n,int offset,int train,int flags,
                 void *rng,next_double uniform,next_uint32 integer,const double *p,
                 const int16_t *forced,int16_t *out,double *audit){
    int xc=initial[0],yc=initial[1],xt=initial[2],yt=initial[3],h=initial[4],s=initial[5];
    for(int i=0;i<n;i++,draws+=3,out+=16){
        int pre=abs(xc-xt)+abs(yc-yt),ox=xt,oy=yt;
        int row=row_of(map,w,xc,yc,xt,yt,h,s),a;
        double eps=p[2]+(p[3]-p[2])*fmin((offset+i)/p[4],1.0);
        if(forced) a=forced[i];
        else if(train && uniform(rng)<eps) a=random_action(rng,integer);
        else a=greedy(q+row);
        if((flags&2)&&a==5&&pre>0)a=4;
        xc=clip(xc+dx[a]);yc=clip(yc+dy[a]);
        int mid=abs(xc-xt)+abs(yc-yt),hit=a==5&&mid==0,miss=a==5&&!hit;
        int turn=draws[0]<p[11],toggle=draws[2]<p[12],bounce=0;
        if(turn)h=(h+(draws[1]<0.5?1:3))%4;
        if(toggle)s=3-s;
        for(int j=0;j<s;j++){
            if(xt+hx[h]!=clip(xt+hx[h])||yt+hy[h]!=clip(yt+hy[h])){h=(h+2)%4;bounce++;}
            xt+=hx[h];yt+=hy[h];
        }
        int post=abs(xc-xt)+abs(yc-yt);
        double base=p[7]+p[8]*hit+p[9]*miss,close=p[10]*(post<=1),pot=p[1]*-post+pre;
        double reward=(base+close)+pot,alpha=0,target=0,before=0,after=0;uint32_t count=0;
        if(train){
            int key=row+a;count=counts[key];
            alpha=p[0]*((flags&1)?pow(1.0+count/p[5],-p[6]):1.0);
            const float *next=q+row_of(map,w,xc,yc,xt,yt,h,s);
            target=reward+p[1]*(double)next[greedy(next)];before=q[key];
            q[key]+=(float)(alpha*(target-before));after=q[key];counts[key]++;
        }
        out[0]=xc;out[1]=yc;out[2]=xt;out[3]=yt;out[4]=h;out[5]=s;
        out[6]=a;out[7]=pre;out[8]=mid;out[9]=post;out[10]=abs(xt-ox)+abs(yt-oy);
        out[11]=hit;out[12]=miss;out[13]=bounce;out[14]=turn;out[15]=toggle;
        if(audit){double v[]={base,close,pot,reward,alpha,target,before,after,(double)count,row/10.0};
            for(int k=0;k<10;k++)audit[i*10+k]=v[k];}
    }
}
