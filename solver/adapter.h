#ifndef MATRIXCORE_ADAPTER_H
#define MATRIXCORE_ADAPTER_H

class Adapter {
public:
    virtual void matmul(const float* A,const float* X,float* Z,int N,int B)=0;
    virtual void add(float* A,const float* B,int size)=0;
    virtual void axpy(float* x,const float* dx,float dt,int size)=0;
    virtual void clamp(float* x,int size,float limit)=0;
    virtual void max_scalar(float* x,int size,float minv)=0;
    virtual void cube(float* out,const float* x,int size)=0;
    virtual ~Adapter() {}
};


#endif //MATRIXCORE_ADAPTER_H
