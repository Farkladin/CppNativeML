//
//  L1L2.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/26/25.
//

#ifndef L1L2_h
#define L1L2_h

#include <cmath>
#include "linAlge/matrix.h"

namespace model{

namespace regularization{

double L1(double loss,linAlge::mat &w,const double lambda=1){
    for(int i=0;i<w.n;i++){
        for(int j=0;j<w.m;j++)loss+=abs(w(i,j));
    }
    return loss;
}

double L2(double loss,linAlge::mat &w,const double lambda=1){
    for(int i=0;i<w.n;i++){
        for(int j=0;j<w.m;j++)loss+=w(i,j)*w(i,j);
    }
    return loss;
}

}

}

#endif /* L1L2_h */
