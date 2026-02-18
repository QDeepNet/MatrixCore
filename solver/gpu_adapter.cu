#include "adapter.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void add_kernel(float* A,const float* B,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) A[i]+=B[i];
}
__global__ void axpy_kernel(float* x,const float* dx,float dt,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=dt*dx[i];
}
__global__ void clamp_kernel(float* x,int n,float lim){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        if(x[i]>lim) x[i]=lim;
        if(x[i]<-lim) x[i]=-lim;
    }
}
__global__ void max_scalar_kernel(float* x,int n,float m){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && x[i]<m) x[i]=m;
}
__global__ void cube_kernel(float* out,const float* x,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=x[i]*x[i]*x[i];
}

class CUDAAdapter : public Adapter {
    cublasHandle_t handle;
public:
    CUDAAdapter(){ cublasCreate(&handle); }
    ~CUDAAdapter(){ cublasDestroy(handle); }

    void matmul(const float* A,const float* X,float* Z,int N,int B) override{
        float a=1.0f,b=0.0f;
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,B,N,&a,A,N,X,N,&b,Z,N);
    }

    void add(float* A,const float* B,int size) override{
        add_kernel<<<(size+255)/256,256>>>(A,B,size);
    }
    void axpy(float* x,const float* dx,float dt,int size) override{
        axpy_kernel<<<(size+255)/256,256>>>(x,dx,dt,size);
    }
    void clamp(float* x,int size,float limit) override{
        clamp_kernel<<<(size+255)/256,256>>>(x,size,limit);
    }
    void max_scalar(float* x,int size,float minv) override{
        max_scalar_kernel<<<(size+255)/256,256>>>(x,size,minv);
    }
    void cube(float* out,const float* x,int size) override{
        cube_kernel<<<(size+255)/256,256>>>(out,x,size);
    }
};
