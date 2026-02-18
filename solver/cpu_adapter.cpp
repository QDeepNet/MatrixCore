#include "adapter.h"
#include <algorithm>

class CPUAdapter : public Adapter {
public:

    void matmul(const float* A,const float* X,float* Z,int N,int B) override {
        for(int i=0;i<N;i++)
            for(int b=0;b<B;b++){
                float s=0;
                for(int k=0;k<N;k++)
                    s+=A[i*N+k]*X[k*B+b];
                Z[i*B+b]=s;
            }
    }

    void add(float* A,const float* B,int size) override {
        for(int i=0;i<size;i++) A[i]+=B[i];
    }

    void axpy(float* x,const float* dx,float dt,int size) override {
        for(int i=0;i<size;i++) x[i]+=dt*dx[i];
    }

    void clamp(float* x,int size,float limit) override {
        for(int i=0;i<size;i++){
            if(x[i]>limit) x[i]=limit;
            if(x[i]<-limit) x[i]=-limit;
        }
    }

    void max_scalar(float* x,int size,float minv) override {
        for(int i=0;i<size;i++)
            if(x[i]<minv) x[i]=minv;
    }

    void cube(float* out,const float* x,int size) override {
        for(int i=0;i<size;i++)
            out[i]=x[i]*x[i]*x[i];
    }
};
