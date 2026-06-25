//
//  reluAct.h
//  CppNativeML
//

#ifndef reluAct_h
#define reluAct_h

namespace model {
namespace actFunction {

inline double relu(double x)  { return x > 0.0 ? x : 0.0; }

// GradientDescent passes activation value a = relu(z) to dactf.
// relu(z) > 0 iff z > 0, so the sign of a recovers the mask.
inline double drelu(double a) { return a > 0.0 ? 1.0 : 0.0; }

} // namespace actFunction
} // namespace model

#endif /* reluAct_h */
