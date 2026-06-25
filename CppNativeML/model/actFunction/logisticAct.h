//
//  logisticAct.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/26/25.
//

#ifndef logisticAct_h
#define logisticAct_h

#include <cmath>

namespace model{

namespace actFunction{

inline double logistic(double x){
    return 1.0 / (1.0 + exp(-x));
}

// Takes activation output a = logistic(z), returns a*(1-a).
// GradientDescent.h passes activation values to dactf, so this matches.
inline double dlogistic(double a){
    return a * (1.0 - a);
}

}

}

#endif /* logisticAct_h */
