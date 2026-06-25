//
//  MSE.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/26/25.
//

#ifndef MSE_h
#define MSE_h

#include "linAlge/matrix.h"
#include <cmath>
#include <vector>

namespace model{

namespace objFunction{

using u32=uint_fast32_t;

inline double MSE(linAlge::mat &h,linAlge::mat &y){
    u32 i,j;
    double ret=0,tmp=0;
    
    assert(("Size of h and y is must be equal.\n"&&(h.n==y.n&&h.m==y.m)));
    for(j=0;j<h.m;j++){
        tmp=0;
        for(i=0;i<h.n;i++){
            tmp+=(h(i,j)-y(i,j))*(h(i,j)-y(i,j));
        }
        ret+=tmp/double(h.n);
    }
    
    return ret;
}

template<class Act>
double calcMSE(linAlge::mat &X,std::vector<linAlge::mat>&W,linAlge::mat &y,Act actf){
    u32 i,j;
    for(linAlge::mat &w:W){
        X=X*w;
        for(i=0;i<X.n;i++){
            for(j=0;j<X.m;j++)X(i,j)=actf(X(i,j));
        }
    }
    return MSE(X,y);
}

inline linAlge::mat dMSE(linAlge::mat &h,linAlge::mat &y){
    u32 i,j;
    linAlge::mat ret(h.n,h.m);
    const double inv_n = 2.0 / h.n;   // MSE averages over n; gradient must match
    for(i=0;i<h.n;i++){
        for(j=0;j<h.m;j++){
            ret(i,j)=inv_n*(h(i,j)-y(i,j));
        }
    }
    return ret;
}

}

}

#endif /* MSE_h */
