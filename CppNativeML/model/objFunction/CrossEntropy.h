//
//  CrossEntropy.h
//  CppNativeML
//

#ifndef CrossEntropy_h
#define CrossEntropy_h

#include "linAlge/matrix.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace model {
namespace objFunction {

// ── Softmax (row-wise, numerically stable) ────────────────────────────────────
// Subtracts row-max before exp() to prevent overflow.
// Applied to the output layer when using categorical CE.
inline void softmax_rows(linAlge::mat& x) {
    for (int i = 0; i < x.n; i++) {
        double* row = x.row(i);
        double maxv = *std::max_element(row, row + x.m);
        double sum  = 0.0;
        for (int j = 0; j < x.m; j++) { row[j] = std::exp(row[j] - maxv); sum += row[j]; }
        for (int j = 0; j < x.m; j++)   row[j] /= sum;
    }
}

// ── Categorical cross-entropy (softmax output) ────────────────────────────────
// L = -(1/n) * sum_i log(h_i[correct_class])
// Only the Y=1 term contributes (one-hot encoding).
inline double CE_cat(linAlge::mat& h, linAlge::mat& y) {
    constexpr double eps = 1e-12;
    double ret = 0.0;
    for (int i = 0; i < h.n; i++)
        for (int j = 0; j < h.m; j++)
            if (y(i,j) > 0.5)
                ret -= std::log(h(i,j) + eps);
    return ret / h.n;
}

// Forward pass + categorical CE (softmax at output, actf at hidden layers).
template<class Act>
double calcCE_cat(linAlge::mat& X, std::vector<linAlge::mat>& W,
                  linAlge::mat& y, Act hidden_actf) {
    for (std::size_t idx = 0; idx < W.size(); idx++) {
        X = X * W[idx];
        if (idx == W.size() - 1) {
            softmax_rows(X);
        } else {
            for (int i = 0; i < X.n; i++)
                for (int j = 0; j < X.m; j++)
                    X(i,j) = hidden_actf(X(i,j));
        }
    }
    return CE_cat(X, y);
}

// ── Binary cross-entropy (sigmoid output, 10 independent binary classifiers) ──
// L = -(1/n) * sum_i sum_j [ Y_ij*log(h_ij) + (1-Y_ij)*log(1-h_ij) ]
inline double CE(linAlge::mat& h, linAlge::mat& y) {
    constexpr double eps = 1e-12;
    double ret = 0.0;
    for (int i = 0; i < h.n; i++)
        for (int j = 0; j < h.m; j++)
            ret -= y(i,j) * std::log(h(i,j) + eps)
                 + (1.0 - y(i,j)) * std::log(1.0 - h(i,j) + eps);
    return ret / h.n;
}

template<class Act>
double calcCE(linAlge::mat& X, std::vector<linAlge::mat>& W,
              linAlge::mat& y, Act actf) {
    for (linAlge::mat& w : W) {
        X = X * w;
        for (int i = 0; i < X.n; i++)
            for (int j = 0; j < X.m; j++)
                X(i,j) = actf(X(i,j));
    }
    return CE(X, y);
}

// ── Combined output gradient: (h - y) / n ────────────────────────────────────
// Works for both:
//   • sigmoid + binary CE:      dBCE/dh ⊙ h*(1-h) = (h-y)/n
//   • softmax + categorical CE: dCatCE/dz           = (h-y)/n
// Pass combined_output_delta=true to gradientDescent so dlogistic is NOT
// additionally applied to this result.
inline linAlge::mat dCE(linAlge::mat& h, linAlge::mat& y) {
    linAlge::mat ret(h.n, h.m);
    const double inv_n = 1.0 / h.n;
    for (int i = 0; i < h.n; i++)
        for (int j = 0; j < h.m; j++)
            ret(i,j) = inv_n * (h(i,j) - y(i,j));
    return ret;
}

} // namespace objFunction
} // namespace model

#endif /* CrossEntropy_h */
