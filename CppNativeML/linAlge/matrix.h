//
//  matrix.h
//  CSE302
//
//  Created by Sora Sugiyama on 3/17/25.
//

#ifndef matrix_h
#define matrix_h

#include <vector>
#include <algorithm>
#include <thread>
#include <cstdint>
#include <cstring>

// ── Cross-platform SIMD detection ────────────────────────────────────────────
#if defined(__AVX2__) && defined(__FMA__)
#   include <immintrin.h>
#   define MAT_AVX2_FMA 1
#elif defined(__AVX2__)
#   include <immintrin.h>
#   define MAT_AVX2 1
#elif defined(__SSE2__)
#   include <emmintrin.h>
#   define MAT_SSE2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#   include <arm_neon.h>
#   define MAT_NEON 1
#endif
// ─────────────────────────────────────────────────────────────────────────────

// ── Z-order (Morton) layout ──────────────────────────────────────────────────
//
// Element (i,j) is stored at index zidx(i,j) = spread(j) | (spread(i) << 1)
// where spread() expands each bit into every other bit position.
//
// Key property: any aligned 2^k × 2^k submatrix starting at (I,J)
// with I,J multiples of 2^k occupies a CONTIGUOUS block of 4^k elements.
//
// Visual for a 4×4 matrix (numbers = storage index of each element):
//
//   ┌─────┬─────┬─────┬─────┐
//   │  0  │  1  │  4  │  5  │   ← two 2×2 Z-shapes in the top half
//   │  2  │  3  │  6  │  7  │
//   ├─────┼─────┼─────┼─────┤
//   │  8  │  9  │ 12  │ 13  │   ← two 2×2 Z-shapes in the bottom half
//   │ 10  │ 11  │ 14  │ 15  │
//   └─────┴─────┴─────┴─────┘
//
// Consequence for D&C on a 2N×2N Z-ordered matrix (N = power of 2):
//   A11 (top-left  N×N) → indices [0,   N²)   ← just pointer + 0
//   A12 (top-right N×N) → indices [N², 2N²)   ← just pointer + N²
//   A21 (bot-left  N×N) → indices [2N², 3N²)  ← just pointer + 2N²
//   A22 (bot-right N×N) → indices [3N², 4N²)  ← just pointer + 3N²
//
// No submatrix extraction loop needed — each quadrant is a plain pointer offset.
// ─────────────────────────────────────────────────────────────────────────────

namespace linAlge {

using lf = double;

class mat {
    using vv = std::vector<std::vector<lf>>;

    // ── Primary storage: flat row-major ──────────────────────────────────────
    // Element (i,j) lives at M[i*m + j].
    // Contiguous rows allow SIMD to operate on a full row with a single pointer.
    std::vector<lf> M;

public:
    int n, m;

    mat(int r = 0, int c = 0) : M(r * c, 0.0), n(r), m(c) {}

    lf&       operator()(int r, int c)       { return M[r * m + c]; }
    const lf& operator()(int r, int c) const { return M[r * m + c]; }

    // Pointer to the start of row r — used by SIMD kernels.
    lf*       row(int r)       { return M.data() + r * m; }
    const lf* row(int r) const { return M.data() + r * m; }

    // NOTE: operator< and operator- have non-standard semantics by design;
    // see README § "Non-standard operator conventions".
    mat& operator<(mat& ref) {
        n = ref.n; m = ref.m; M = ref.M;
        return *this;
    }
    mat& operator=(mat ref) {
        n = ref.n; m = ref.m; M = ref.M;
        return *this;
    }
    mat& operator=(vv ref) {
        n = (int)ref.size();
        m = n ? (int)ref[0].size() : 0;
        M.resize(n * m);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                M[i*m+j] = ref[i][j];
        return *this;
    }

    // In-place negation (non-standard: modifies *this and returns ref to it).
    mat& operator-() {
        for (lf& v : M) v = -v;
        return *this;
    }
    mat operator+(mat ref) const {
        mat ret(n, m);
        for (int k = 0, sz = n*m; k < sz; k++) ret.M[k] = M[k] + ref.M[k];
        return ret;
    }
    mat& operator+=(mat ref) {
        for (int k = 0, sz = n*m; k < sz; k++) M[k] += ref.M[k];
        return *this;
    }
    mat operator-(mat ref) const {
        mat ret(n, m);
        for (int k = 0, sz = n*m; k < sz; k++) ret.M[k] = M[k] - ref.M[k];
        return ret;
    }

    // Blocked transpose (8×8 tiles to stay within a cache-line width)
    mat T() const {
        mat ret(m, n);
        constexpr int TB = 8;
        const int i_end = (m / TB) * TB, j_end = (n / TB) * TB;
        for (int i = 0; i < i_end; i += TB)
            for (int j = 0; j < j_end; j += TB)
                for (int ii = i; ii < i+TB; ii++)
                    for (int jj = j; jj < j+TB; jj++)
                        ret.M[ii*n + jj] = M[jj*m + ii];
        for (int ii = 0; ii < i_end; ii++)
            for (int jj = j_end; jj < n; jj++)
                ret.M[ii*n + jj] = M[jj*m + ii];
        for (int ii = i_end; ii < m; ii++)
            for (int jj = 0; jj < n; jj++)
                ret.M[ii*n + jj] = M[jj*m + ii];
        return ret;
    }

    bool operator==(const mat& ref) const {
        if (n != ref.n || m != ref.m) return false;
        return M == ref.M;
    }
    bool operator==(const vv& ref) const {
        if ((int)ref.size() != n) return false;
        for (int i = 0; i < n; i++) {
            if ((int)ref[i].size() != m) return false;
            for (int j = 0; j < m; j++)
                if (M[i*m+j] != ref[i][j]) return false;
        }
        return true;
    }

    mat  operator*(mat ref) const { return dispatchMul(*this, ref); }
    mat& operator*=(const mat& ref) { *this = dispatchMul(*this, ref); return *this; }

    mat naiveProduct(const mat& a, const mat& b) { return nProduct(a, b); }

private:
    // =========================================================================
    // SIMD axpy: ret_row[j] += aik * b_row[j]  for j in [0, len)
    // Both pointers are row-starts from the flat array → guaranteed contiguous.
    // =========================================================================
    static void axpy(lf* __restrict__ ret_row,
                     const lf* __restrict__ b_row,
                     lf aik, int len) {
        int j = 0;
#if MAT_AVX2_FMA
        __m256d va = _mm256_set1_pd(aik);
        for (; j+4 <= len; j += 4)
            _mm256_storeu_pd(ret_row+j,
                _mm256_fmadd_pd(va, _mm256_loadu_pd(b_row+j),
                                    _mm256_loadu_pd(ret_row+j)));
#elif MAT_AVX2
        __m256d va = _mm256_set1_pd(aik);
        for (; j+4 <= len; j += 4)
            _mm256_storeu_pd(ret_row+j,
                _mm256_add_pd(_mm256_loadu_pd(ret_row+j),
                              _mm256_mul_pd(va, _mm256_loadu_pd(b_row+j))));
#elif MAT_SSE2
        __m128d va = _mm_set1_pd(aik);
        for (; j+2 <= len; j += 2)
            _mm_storeu_pd(ret_row+j,
                _mm_add_pd(_mm_loadu_pd(ret_row+j),
                           _mm_mul_pd(va, _mm_loadu_pd(b_row+j))));
#elif MAT_NEON
        float64x2_t va = vdupq_n_f64(aik);
        for (; j+2 <= len; j += 2)
            vst1q_f64(ret_row+j,
                vmlaq_f64(vld1q_f64(ret_row+j), vld1q_f64(b_row+j), va));
#endif
        for (; j < len; j++) ret_row[j] += aik * b_row[j];
    }

    // =========================================================================
    // Tiled ikj multiply: three TI×TK tiles fit in M3 Pro's 128KB L1 cache.
    // Parallelised across the i-dimension with std::thread.
    // =========================================================================
    static mat nProduct(const mat& A, const mat& B) {
        const int N = A.n, K = A.m, Mc = B.m;
        mat ret(N, Mc);
        constexpr int TI = 48, TK = 48;

        auto worker = [&](int i0, int i1) {
            for (int ii = i0; ii < i1; ii += TI) {
                const int ie = std::min(ii+TI, i1);
                for (int kk = 0; kk < K; kk += TK) {
                    const int ke = std::min(kk+TK, K);
                    for (int i = ii; i < ie; i++) {
                        lf* rrow = ret.row(i);
                        for (int k = kk; k < ke; k++) {
                            const lf aik = A.M[i*K + k];
                            if (aik == 0.0) continue;
                            axpy(rrow, B.row(k), aik, Mc);
                        }
                    }
                }
            }
        };

        const int hw = (int)std::thread::hardware_concurrency();
        const int nt = (N*K*Mc >= (1<<16) && hw > 1) ? std::min(N, hw) : 1;
        if (nt == 1) {
            worker(0, N);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(nt);
            const int chunk = (N + nt - 1) / nt;
            for (int t = 0; t < nt; t++)
                threads.emplace_back(worker, t*chunk, std::min((t+1)*chunk, N));
            for (auto& th : threads) th.join();
        }
        return ret;
    }

    // =========================================================================
    // Z-order helpers
    // =========================================================================

    // Spread 16 bits to even bit positions of a 32-bit word.
    // Input bit k → output bit 2k.  Supports row/col indices up to 65535.
    static uint32_t spread16(uint32_t x) {
        x &= 0x0000ffff;
        x = (x ^ (x <<  8)) & 0x00ff00ff;
        x = (x ^ (x <<  4)) & 0x0f0f0f0f;
        x = (x ^ (x <<  2)) & 0x33333333;
        x = (x ^ (x <<  1)) & 0x55555555;
        return x;
    }
    // Morton code: col bits in even positions, row bits in odd positions.
    static uint32_t zidx(uint32_t row, uint32_t col) {
        return spread16(col) | (spread16(row) << 1);
    }
    static int nextPow2(int x) { int p=1; while(p<x) p<<=1; return p; }

    // Pack row-major mat into a P×P Z-order flat array (P = nextPow2(N)).
    // Elements outside [0,A.n)×[0,A.m) remain 0 (zero-padding).
    static void zpack(const mat& A, lf* buf, int /*P*/) {
        for (int i = 0; i < A.n; i++)
            for (int j = 0; j < A.m; j++)
                buf[zidx(i,j)] = A.M[i*A.m + j];
    }
    // Unpack P×P Z-order array back to a rows×cols row-major mat.
    static mat zunpack(const lf* buf, int rows, int cols) {
        mat ret(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ret.M[i*cols + j] = buf[zidx(i,j)];
        return ret;
    }

    // ── Element-wise SIMD kernels for Z-order flat buffers ───────────────────
    // All three pointers must be distinct (no aliasing).
    static void zadd(const lf* __restrict__ A, const lf* __restrict__ B,
                     lf* __restrict__ C, int n) {
        int k = 0;
#if MAT_AVX2_FMA || MAT_AVX2
        for (; k+4 <= n; k += 4)
            _mm256_storeu_pd(C+k, _mm256_add_pd(_mm256_loadu_pd(A+k),
                                                 _mm256_loadu_pd(B+k)));
#elif MAT_SSE2
        for (; k+2 <= n; k += 2)
            _mm_storeu_pd(C+k, _mm_add_pd(_mm_loadu_pd(A+k), _mm_loadu_pd(B+k)));
#elif MAT_NEON
        for (; k+2 <= n; k += 2)
            vst1q_f64(C+k, vaddq_f64(vld1q_f64(A+k), vld1q_f64(B+k)));
#endif
        for (; k < n; k++) C[k] = A[k] + B[k];
    }
    static void zsub(const lf* __restrict__ A, const lf* __restrict__ B,
                     lf* __restrict__ C, int n) {
        int k = 0;
#if MAT_AVX2_FMA || MAT_AVX2
        for (; k+4 <= n; k += 4)
            _mm256_storeu_pd(C+k, _mm256_sub_pd(_mm256_loadu_pd(A+k),
                                                 _mm256_loadu_pd(B+k)));
#elif MAT_SSE2
        for (; k+2 <= n; k += 2)
            _mm_storeu_pd(C+k, _mm_sub_pd(_mm_loadu_pd(A+k), _mm_loadu_pd(B+k)));
#elif MAT_NEON
        for (; k+2 <= n; k += 2)
            vst1q_f64(C+k, vsubq_f64(vld1q_f64(A+k), vld1q_f64(B+k)));
#endif
        for (; k < n; k++) C[k] = A[k] - B[k];
    }

    // =========================================================================
    // winograd_z: recursive Strassen-Winograd on Z-order buffers.
    //
    // A, B:  N×N Z-ordered inputs  (N must be a power of 2).
    // C:     N×N Z-ordered output  (fully overwritten; need not be pre-zeroed).
    // ws:    scratch workspace of at least 15·(N/2)² doubles, carved level by
    //        level; caller passes the same ws+15h slice to all 7 sub-calls
    //        (they are sequential, so the same memory is safely reused).
    //
    // Total workspace required by the outer call:
    //   W(N) = 15·(N/2)² + W(N/2) → 5·N² doubles  (geometric series).
    //
    // Because of the Z-order property the four N/2×N/2 quadrants of a 2N×2N
    // buffer lie at pointer offsets {0, h, 2h, 3h} (h = (N/2)²) — zero-copy.
    // =========================================================================
    static void winograd_z(const lf* A, const lf* B, lf* C, int N, lf* ws) {
        if (N == 1) { C[0] = A[0] * B[0]; return; }

        const int M = N >> 1;
        const int h = M * M;          // elements per quadrant

        // ── Carve 15 scratch buffers from ws ─────────────────────────────────
        lf *S1=ws,    *S2=ws+h,  *S3=ws+2*h, *S4=ws+3*h;
        lf *m1=ws+4*h, *m2=ws+5*h, *m3=ws+6*h, *m4=ws+7*h,
           *m5=ws+8*h, *m6=ws+9*h, *m7=ws+10*h;
        lf *t1=ws+11*h, *t2=ws+12*h, *T1=ws+13*h, *T2=ws+14*h;
        lf *next = ws + 15*h;         // sub-calls share this remainder

        // ── Zero-copy quadrant split (Z-order property) ───────────────────────
        const lf *A11=A,   *A12=A+h, *A21=A+2*h, *A22=A+3*h;
        const lf *B11=B,   *B12=B+h, *B21=B+2*h, *B22=B+3*h;
        lf       *C11=C,   *C12=C+h, *C21=C+2*h, *C22=C+3*h;

        // ── Strassen pre-additions ────────────────────────────────────────────
        zadd(A21,A22,S1,h);  zsub(S1,A11,S2,h);
        zsub(B12,B11,S3,h);  zsub(B22,S3,S4,h);

        // ── 7 recursive multiplications ───────────────────────────────────────
        winograd_z(S2, S4, m1, M, next);
        winograd_z(A11, B11, m2, M, next);
        winograd_z(A12, B21, m3, M, next);
        zsub(A11,A21,t1,h); zsub(B22,B12,t2,h);
        winograd_z(t1, t2, m4, M, next);
        winograd_z(S1, S3, m5, M, next);
        zsub(A12,S2,t1,h);
        winograd_z(t1, B22, m6, M, next);
        zsub(S4,B21,t2,h);
        winograd_z(A22, t2, m7, M, next);

        // ── Assemble result quadrants ─────────────────────────────────────────
        zadd(m1,m2,T1,h);
        zadd(T1,m4,T2,h);

        zadd(m2,m3,C11,h);                  // C11 = M2+M3
        // C12 = T1+M5+M6 — use S1 (free after recursive calls) to avoid aliasing
        zadd(T1,m5,S1,h); zadd(S1,m6,C12,h);
        zsub(T2,m7,C21,h);                  // C21 = T2-M7
        zadd(T2,m5,C22,h);                  // C22 = T2+M5
    }

    // =========================================================================
    // Winograd entry: pack row-major → Z-order, recurse with pre-allocated
    // workspace (5·P² doubles), unpack.  Handles any square N (odd or even)
    // by padding to the next power of 2.
    // =========================================================================
    static mat Winograd(const mat& A, const mat& B) {
        const int N = A.n;
        const int P = nextPow2(N);

        // Za, Zb, Zc: P×P Z-order buffers.
        // ws: scratch for winograd_z — geometric series gives W(P) = 5·P² doubles.
        std::vector<lf> Za(P*P, 0.0), Zb(P*P, 0.0), Zc(P*P, 0.0);
        std::vector<lf> ws(5*P*P);
        zpack(A, Za.data(), P);
        zpack(B, Zb.data(), P);

        winograd_z(Za.data(), Zb.data(), Zc.data(), P, ws.data());

        return zunpack(Zc.data(), N, N);
    }

    // =========================================================================
    // Block D&C for non-square matrices.  Uses memcpy for row-major submatrix
    // extraction (one call per row vs element-by-element).
    // =========================================================================
    static mat product(const mat& A, const mat& B) {
        const int N = A.n, K = A.m, Mc = B.m;
        int t = std::min({N, K, Mc});
        if (t & 1) t ^= 1;
        if (t == 0) return nProduct(A, B);

        mat A11(t,t), A12(t,K-t), A21(N-t,t), A22(N-t,K-t);
        mat B11(t,t), B12(t,Mc-t), B21(K-t,t), B22(K-t,Mc-t);

        for (int i = 0; i < t; i++) {
            std::memcpy(A11.row(i), A.row(i),       t    * sizeof(lf));
            std::memcpy(A12.row(i), A.row(i) + t,  (K-t) * sizeof(lf));
        }
        for (int i = 0; i < N-t; i++) {
            std::memcpy(A21.row(i), A.row(i+t),       t    * sizeof(lf));
            std::memcpy(A22.row(i), A.row(i+t) + t,  (K-t) * sizeof(lf));
        }
        for (int i = 0; i < t; i++) {
            std::memcpy(B11.row(i), B.row(i),       t     * sizeof(lf));
            std::memcpy(B12.row(i), B.row(i) + t,  (Mc-t) * sizeof(lf));
        }
        for (int i = 0; i < K-t; i++) {
            std::memcpy(B21.row(i), B.row(i+t),       t     * sizeof(lf));
            std::memcpy(B22.row(i), B.row(i+t) + t,  (Mc-t) * sizeof(lf));
        }

        mat C11 = A11*B11 + A12*B21;
        mat C12 = A11*B12 + A12*B22;
        mat C21 = A21*B11 + A22*B21;
        mat C22 = A21*B12 + A22*B22;

        mat ret(N, Mc);
        for (int i = 0; i < t; i++) {
            std::memcpy(ret.row(i),     C11.row(i),  t     * sizeof(lf));
            std::memcpy(ret.row(i)+t,   C12.row(i), (Mc-t) * sizeof(lf));
        }
        for (int i = 0; i < N-t; i++) {
            std::memcpy(ret.row(i+t),   C21.row(i),  t     * sizeof(lf));
            std::memcpy(ret.row(i+t)+t, C22.row(i), (Mc-t) * sizeof(lf));
        }
        return ret;
    }

    // Dispatch: nProduct for small dims, Winograd for large square, D&C otherwise.
    // !(N & 1) guard removed — Winograd now pads to nextPow2 internally.
    static mat dispatchMul(const mat& A, const mat& B) {
        const int N = A.n, K = A.m, Mc = B.m;
        if (N <= 180 || K <= 180 || Mc <= 180) return nProduct(A, B);
        if (N == K && K == Mc)                 return Winograd(A, B);
        return product(A, B);
    }
};

} // namespace linAlge

#endif /* matrix_h */
