//
//  GradientDescent.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/26/25.
//  Fixed & optimized: backprop matrix bug fix + cross-platform SIMD + mini-batch
//

#ifndef GradientDescent_h
#define GradientDescent_h

#include "linAlge/matrix.h"
#include <vector>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <stack>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

// ── Cross-platform SIMD detection ────────────────────────────────────────────
#if defined(__AVX2__) && defined(__FMA__)
#   include <immintrin.h>
#   define SIMD_AVX2_FMA 1
#elif defined(__AVX2__)
#   include <immintrin.h>
#   define SIMD_AVX2 1
#elif defined(__SSE2__)
#   include <emmintrin.h>
#   define SIMD_SSE2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#   include <arm_neon.h>
#   define SIMD_NEON 1
#endif
// ─────────────────────────────────────────────────────────────────────────────

namespace optimizer {

using u32 = uint_fast32_t;

// dst[i] *= src[i]
static inline void mulElemInPlace(double* __restrict__ dst,
                                  const double* __restrict__ src,
                                  u32 len) {
    u32 i = 0;
#if SIMD_AVX2_FMA || SIMD_AVX2
    for (; i + 4 <= len; i += 4)
        _mm256_storeu_pd(dst+i, _mm256_mul_pd(_mm256_loadu_pd(dst+i), _mm256_loadu_pd(src+i)));
#elif SIMD_SSE2
    for (; i + 2 <= len; i += 2)
        _mm_storeu_pd(dst+i, _mm_mul_pd(_mm_loadu_pd(dst+i), _mm_loadu_pd(src+i)));
#elif SIMD_NEON
    for (; i + 2 <= len; i += 2)
        vst1q_f64(dst+i, vmulq_f64(vld1q_f64(dst+i), vld1q_f64(src+i)));
#endif
    for (; i < len; i++) dst[i] *= src[i];
}

// dst[i] *= scale  (L2 weight decay)
static inline void scaleInPlace(double* dst, double scale, u32 len) {
    u32 i = 0;
#if SIMD_AVX2_FMA || SIMD_AVX2
    __m256d vs = _mm256_set1_pd(scale);
    for (; i + 4 <= len; i += 4)
        _mm256_storeu_pd(dst+i, _mm256_mul_pd(_mm256_loadu_pd(dst+i), vs));
#elif SIMD_SSE2
    __m128d vs = _mm_set1_pd(scale);
    for (; i + 2 <= len; i += 2)
        _mm_storeu_pd(dst+i, _mm_mul_pd(_mm_loadu_pd(dst+i), vs));
#elif SIMD_NEON
    float64x2_t vs = vdupq_n_f64(scale);
    for (; i + 2 <= len; i += 2)
        vst1q_f64(dst+i, vmulq_f64(vld1q_f64(dst+i), vs));
#endif
    for (; i < len; i++) dst[i] *= scale;
}

// dst[i] -= scale * src[i]
static inline void subScaledInPlace(double* __restrict__ dst,
                                    const double* __restrict__ src,
                                    double scale, u32 len) {
    u32 i = 0;
#if SIMD_AVX2_FMA
    __m256d vscale = _mm256_set1_pd(scale);
    for (; i + 4 <= len; i += 4)
        _mm256_storeu_pd(dst+i, _mm256_fnmadd_pd(vscale, _mm256_loadu_pd(src+i),
                                                           _mm256_loadu_pd(dst+i)));
#elif SIMD_AVX2
    __m256d vscale = _mm256_set1_pd(scale);
    for (; i + 4 <= len; i += 4)
        _mm256_storeu_pd(dst+i, _mm256_sub_pd(_mm256_loadu_pd(dst+i),
                                              _mm256_mul_pd(vscale, _mm256_loadu_pd(src+i))));
#elif SIMD_SSE2
    __m128d vscale = _mm_set1_pd(scale);
    for (; i + 2 <= len; i += 2)
        _mm_storeu_pd(dst+i, _mm_sub_pd(_mm_loadu_pd(dst+i),
                                         _mm_mul_pd(vscale, _mm_loadu_pd(src+i))));
#elif SIMD_NEON
    float64x2_t vscale = vdupq_n_f64(scale);
    for (; i + 2 <= len; i += 2)
        vst1q_f64(dst+i, vsubq_f64(vld1q_f64(dst+i),
                                    vmulq_f64(vscale, vld1q_f64(src+i))));
#endif
    for (; i < len; i++) dst[i] -= scale * src[i];
}

// delta[r][c] *= dactf(ref[r][c])
template<class DAct>
static inline void applyDActInPlace(linAlge::mat& delta,
                                    const linAlge::mat& ref,
                                    DAct dactf) {
    const int rows = delta.n, cols = delta.m;
    for (int r = 0; r < rows; r++) {
        double*       d  = &delta(r, 0);
        const double* rf = &ref(r, 0);
        for (int c = 0; c < cols; c++)
            d[c] *= dactf(rf[c]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gradientDescent
//
// Parameters:
//   combined_output_delta  when true, dloss already returns dL/dz at the
//                          output (loss+activation gradients pre-combined).
//                          Required for CE+sigmoid or CE+softmax so that
//                          dactf is NOT re-applied to the output delta.
//   lambda                 L2 regularization: W *= (1-theta*lambda) per step.
//   use_softmax_output     apply numerically-stable row-wise softmax to the
//                          last layer instead of element-wise actf.
//   batch_size             0 = full-batch; >0 = mini-batch SGD with this
//                          many samples per gradient step. Indices are
//                          shuffled once per epoch (when all samples used).
// ─────────────────────────────────────────────────────────────────────────────
template<class F, class DF, class Act, class DAct>
void gradientDescent(
    F        loss,
    DF       dloss,
    Act      actf,
    DAct     dactf,
    std::vector<linAlge::mat>& coef,
    linAlge::mat& X,
    linAlge::mat& Y,
    double theta,
    u32    gn                    = 10000,
    u32    printCur              = 1u << 30,
    u32    step_offset           = 0,
    bool   combined_output_delta = false,
    double lambda                = 0.0,
    bool   use_softmax_output    = false,
    int    batch_size            = 0)
{
    const double decay = 1.0 - theta * lambda;
    const int    n     = X.n;
    const bool   do_mb = (batch_size > 0 && batch_size < n);
    const int    bs    = do_mb ? batch_size : n;

    // ── Permutation for mini-batch shuffling ──────────────────────────────
    std::mt19937 rng(42);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    int batch_pos = n;   // set to n so first iteration triggers a shuffle

    // ── Pre-allocate batch buffers ────────────────────────────────────────
    // For full-batch: Xb/Yb are one-time copies of X/Y (same as the original
    // `linAlge::mat x = X` copy that happened inside the loop before).
    // For mini-batch: only bs rows are copied per step.
    linAlge::mat Xb(bs, X.m), Yb(bs, Y.m);
    if (!do_mb) { Xb = X; Yb = Y; }

    for (u32 step = 1; step <= gn; step++) {

        // ── Fill mini-batch ───────────────────────────────────────────────
        if (do_mb) {
            if (batch_pos + bs > n) {
                std::shuffle(perm.begin(), perm.end(), rng);
                batch_pos = 0;
            }
            for (int i = 0; i < bs; i++) {
                const int r = perm[batch_pos + i];
                std::memcpy(Xb.row(i), X.row(r), X.m * sizeof(double));
                std::memcpy(Yb.row(i), Y.row(r), Y.m * sizeof(double));
            }
            batch_pos += bs;
        }

        // ── Forward pass ──────────────────────────────────────────────────
        std::stack<linAlge::mat> log;
        linAlge::mat x = Xb;

        for (std::size_t li = 0; li < coef.size(); li++) {
            log.push(x);
            x = x * coef[li];
            if (use_softmax_output && li == coef.size() - 1) {
                for (int ii = 0; ii < x.n; ii++) {
                    double* row = &x(ii, 0);
                    double maxv = *std::max_element(row, row + x.m);
                    double sum  = 0.0;
                    for (int jj = 0; jj < x.m; jj++) { row[jj] = std::exp(row[jj]-maxv); sum += row[jj]; }
                    for (int jj = 0; jj < x.m; jj++)   row[jj] /= sum;
                }
            } else {
                for (int ii = 0; ii < x.n; ii++)
                    for (int jj = 0; jj < x.m; jj++)
                        x(ii, jj) = actf(x(ii, jj));
            }
        }

        // Print: for mini-batch, loss is over the current batch (noisy but fast).
        if ((step - 1) % printCur == 0)
            std::cout << std::fixed << std::setprecision(6)
                      << "step " << step_offset + step - 1
                      << "  loss: " << loss(x, Yb) << "\n";

        // ── Output delta ──────────────────────────────────────────────────
        linAlge::mat a_last = x;
        x = dloss(a_last, Yb);
        if (!combined_output_delta)
            applyDActInPlace(x, a_last, dactf);

        // ── Backward pass ─────────────────────────────────────────────────
        for (auto it = coef.rbegin(); it != coef.rend(); it++) {
            linAlge::mat& w     = *it;
            linAlge::mat a_prev = log.top(); log.pop();

            linAlge::mat dW = a_prev.T() * x;

            if (std::next(it) != coef.rend()) {
                x = x * w.T();
                applyDActInPlace(x, a_prev, dactf);
            }

            if (lambda > 0.0)
                for (int r = 0; r < w.n; r++)
                    scaleInPlace(&w(r, 0), decay, static_cast<u32>(w.m));

            for (int r = 0; r < w.n; r++)
                subScaledInPlace(&w(r, 0), &dW(r, 0), theta,
                                 static_cast<u32>(w.m));
        }
    }
}

} // namespace optimizer

#endif /* GradientDescent_h */
