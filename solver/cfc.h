#ifndef MATRIXCORE_CFC_H
#define MATRIXCORE_CFC_H

#include "adapter.h"
#include <vector>
#include <cmath>
#include <random>

class CFC {
public:
    int N,B,n_iter;
    float dt=0.1f;
    float beta=0.15f;
    float alpha=1.0f;
    float xi;

    std::vector<float> J,x,e,z,dx,tmp;
    Adapter* backend;

    CFC(int N_,int B_,int it,const std::vector<float>& J_,Adapter* a)
    :N(N_),B(B_),n_iter(it),J(J_),backend(a)
    {
        x.resize(N*B);
        e.assign(N*B,1.0f);
        z.resize(N*B);
        dx.resize(N*B);
        tmp.resize(N*B);

        float s=0;
        for(float v:J) s+=v*v;
        if(s<1e-8f) s=1e-8f;
        xi=std::sqrt(2.0f*N/s);

        std::mt19937 g(1);
        std::normal_distribution<float>d(0,0.1);
        for(float&v:x) v=d(g);
    }

    void run(){
        for(int it=0;it<n_iter;it++){

            backend->matmul(J.data(),x.data(),z.data(),N,B);

            backend->cube(tmp.data(),x.data(),N*B);

            for(int i=0;i<N*B;i++)
                dx[i]=(-tmp[i]+z[i]*xi*e[i])*dt;

            backend->axpy(x.data(),dx.data(),1.0f,N*B);

            backend->clamp(x.data(),N*B,1.5f);
            backend->max_scalar(e.data(),N*B,0.01f);
        }
    }
};


#endif //MATRIXCORE_CFC_H
