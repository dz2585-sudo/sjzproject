/* Shared-expectation stochastic feedback DP. All ten actions are retained.
 * No RNG or scenario identifier is present in any planning function.
 * Float64 arithmetic, no fast-math/contraction, task reward without shaping. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#define API __declspec(dllexport)
static int64_t now_ns(void){LARGE_INTEGER t,f;QueryPerformanceCounter(&t);QueryPerformanceFrequency(&f);return (int64_t)((double)t.QuadPart*1e9/(double)f.QuadPart);}
#else
#include <time.h>
#define API __attribute__((visibility("default")))
static int64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000000000LL+t.tv_nsec;}
#endif
static const int dx[]={-1,1,0,0,0,0,-3,3,0,0},dy[]={0,0,-1,1,0,0,0,0,-3,3};
static const int hx[]={1,0,-1,0},hy[]={0,1,0,-1};
static int clip(int x){return x<0?0:x>24?24:x;}
static int advance(int *x,int *y,int *h,int s){int b=0;for(int j=0;j<s;j++){
    if(*x+hx[*h]!=clip(*x+hx[*h])||*y+hy[*h]!=clip(*y+hy[*h])){*h=(*h+2)%4;b++;}
    *x+=hx[*h];*y+=hy[*h];}return b;}
static int argmax(const float *q){int b=0;for(int a=1;a<10;a++)if(q[a]>q[b])b=a;return b;}
static int physical_id(int xc,int yc,int xt,int yt,int h,int s){return ((xc*25+yc)*625+xt*25+yt)*8+h*2+s-1;}
static int relative_id(int xc,int yc,int xt,int yt,int h,int s){return (((xt-xc+24)*49+yt-yc+24)*4+h)*20+(s-1)*10;}
API void initialize(int terminal,const float *q,double *value){
    for(int c=0;c<625;c++)for(int t=0;t<5000;t++){
        int xc=c/25,yc=c%25,xt=t/8/25,yt=t/8%25,h=t/2%4,s=t%2+1;
        double v=terminal==1?0.0:-(double)(abs(xc-xt)+abs(yc-yt));
        if(terminal==2){const float *row=q+relative_id(xc,yc,xt,yt,h,s);v+=(double)row[argmax(row)];}
        value[c*5000+t]=v;
    }
}
API void sweep(const double *previous,const int32_t *next,const double *prob,const uint8_t *branches,
               const int16_t *moves,const double *p,double *expected,double *value,uint8_t *policy,uint64_t *stats){
    for(int c=0;c<625;c++)for(int t=0;t<5000;t++){
        double e=0;int x=c/25,y=c%25;
        for(int k=0;k<branches[t];k++){
            int nt=next[t*6+k],d=abs(x-nt/8/25)+abs(y-nt/8%25);
            e+=prob[t*6+k]*(p[1]*(d<=1)+p[0]*previous[c*5000+nt]);stats[1]++;
        }
        expected[c*5000+t]=e;
    }
    for(int c=0;c<625;c++)for(int t=0;t<5000;t++){
        double q[10],best=-INFINITY;
        for(int a=0;a<10;a++){
            double base=p[2];if(a==5)base+=(c==t/8?p[3]:p[4]);
            q[a]=base+expected[moves[c*10+a]*5000+t];if(q[a]>best)best=q[a];stats[2]++;
        }
        int chosen=0;while(q[chosen]<best-p[5])chosen++;
        value[c*5000+t]=best;policy[c*5000+t]=(uint8_t)chosen;stats[0]++;
    }
    stats[3]=stats[2]-3125000u;
}
static int rule(int xc,int yc,int xt,int yt,int h,int s){
    if(xc==xt&&yc==yt)return 5;advance(&xt,&yt,&h,s);
    int best=0,bd=100,bm=100;for(int a=0;a<10;a++){if(a==5)continue;
        int x=clip(xc+dx[a]),y=clip(yc+dy[a]),d=abs(x-xt)+abs(y-yt),m=abs(x-xc)+abs(y-yc);
        if(d<bd||(d==bd&&m<bm)){best=a;bd=d;bm=m;}}
    return best;
}
/* Evaluation owns the random draws. The decision reads only current state and
 * an immutable policy/Q table. Target randomness is consumed after the action. */
API void rollout(const int32_t *initial,const double *draws,int n,int mode,const void *data,
                 const double *p,const double *environment,int16_t *out,int64_t *timing,void *reserved){
    int xc=initial[0],yc=initial[1],xt=initial[2],yt=initial[3],h=initial[4],s=initial[5];
    (void)reserved;
    for(int i=0;i<n;i++,out+=18,draws+=3){
        int pre=abs(xc-xt)+abs(yc-yt),ox=xt,oy=yt,observable_speed=s;
        int64_t start=now_ns();int proposed;
        if(mode==0)proposed=rule(xc,yc,xt,yt,h,s);
        else if(mode==1){const float *q=(const float *)data+relative_id(xc,yc,xt,yt,h,s);proposed=argmax(q);}
        else proposed=((const uint8_t *)data)[physical_id(xc,yc,xt,yt,h,s)];
        int a=(proposed==5&&pre>0)?4:proposed;timing[i]=now_ns()-start;
        xc=clip(xc+dx[a]);yc=clip(yc+dy[a]);int mid=abs(xc-xt)+abs(yc-yt),hit=a==5&&mid==0,miss=a==5&&!hit;
        int turn=draws[0]<environment[0],toggle=draws[2]<environment[1];
        if(turn)h=(h+(draws[1]<.5?1:3))%4;if(toggle)s=3-s;
        int bounce=advance(&xt,&yt,&h,s),post=abs(xc-xt)+abs(yc-yt);
        out[0]=xc;out[1]=yc;out[2]=xt;out[3]=yt;out[4]=h;out[5]=s;out[6]=a;out[7]=pre;
        out[8]=mid;out[9]=post;out[10]=abs(xt-ox)+abs(yt-oy);out[11]=hit;out[12]=miss;out[13]=bounce;
        out[14]=turn;out[15]=toggle;out[16]=proposed;out[17]=observable_speed;
    }
}
