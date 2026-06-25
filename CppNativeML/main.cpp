//
//  main.cpp
//  CSE302
//
//  Created by Sora Sugiyama on 3/17/25.
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "linAlge/matrix.h"
#include "MNIST/readMNIST.h"
#include "optimizer/GradientDescent.h"
#include "model/objFunction/CrossEntropy.h"
#include "model/actFunction/logisticAct.h"
#include "model/actFunction/reluAct.h"

// __FILE__ is the absolute path of this source file at compile time.
static std::string mnistDir() {
    std::string src = __FILE__;
    auto slash = src.rfind('/');
    return src.substr(0, slash + 1) + "MNIST/archive/";
}

static linAlge::mat makeX(const std::vector<std::pair<MNIST::mpix, uint32_t>>& data,
                          int rows) {
    linAlge::mat X(rows, 784);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < 784; j++)
            X(i, j) = static_cast<unsigned char>(data[i].first[j]) / 255.0;
    return X;
}

static linAlge::mat makeY(const std::vector<std::pair<MNIST::mpix, uint32_t>>& data,
                          int rows) {
    linAlge::mat Y(rows, 10);
    for (int i = 0; i < rows; i++)
        Y(i, data[i].second % 10) = 1.0;
    return Y;
}

// Xavier uniform init — for sigmoid/softmax output layers.
// He uniform init — for ReLU hidden layers (limit = sqrt(2/fan_in)).
static linAlge::mat makeWeight(int rows, int cols, uint32_t seed = 42,
                               bool he_init = false) {
    linAlge::mat W(rows, cols);
    std::mt19937 rng(seed);
    const double limit = he_init
        ? std::sqrt(2.0 / static_cast<double>(rows))
        : 1.0 / std::sqrt(static_cast<double>(rows));
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            W(i, j) = dist(rng);
    return W;
}

// Forward pass: ReLU for hidden layers, softmax for the output layer.
static linAlge::mat predict(linAlge::mat X, const std::vector<linAlge::mat>& W) {
    for (std::size_t idx = 0; idx < W.size(); idx++) {
        X = X * W[idx];
        if (idx == W.size() - 1) {
            model::objFunction::softmax_rows(X);
        } else {
            for (int i = 0; i < X.n; i++)
                for (int j = 0; j < X.m; j++)
                    X(i, j) = model::actFunction::relu(X(i, j));
        }
    }
    return X;
}

static int argmax(const linAlge::mat& m, int row) {
    int best = 0;
    for (int j = 1; j < m.m; j++)
        if (m(row, j) > m(row, best)) best = j;
    return best;
}

static void evaluate(const char* tag,
                     const std::vector<std::pair<MNIST::mpix, uint32_t>>& data,
                     const std::vector<linAlge::mat>& W) {
    const int n = (int)data.size();
    linAlge::mat X = makeX(data, n);
    linAlge::mat prob = predict(X, W);

    int correct = 0;
    int classCorrect[10] = {}, classTotal[10] = {};
    for (int i = 0; i < n; i++) {
        int pred  = argmax(prob, i);
        int label = (int)(data[i].second % 10);
        bool ok   = (pred == label);
        correct             += ok;
        classCorrect[label] += ok;
        classTotal[label]++;
    }
    std::cout << "\n[" << tag << "]\n";
    for (int c = 0; c < 10; c++)
        std::cout << "  digit " << c << ": "
                  << classCorrect[c] << "/" << classTotal[c]
                  << "  (" << 100.0 * classCorrect[c] / classTotal[c] << "%)\n";
    std::cout << "  total accuracy: " << correct << "/" << n
              << "  (" << 100.0 * correct / n << "%)\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const std::string imgDir = mnistDir();

    auto train = MNIST::readMNIST(imgDir + "train-images.idx3-ubyte",
                                  imgDir + "train-labels.idx1-ubyte");   // 60,000
    auto test  = MNIST::readMNIST(imgDir + "t10k-images.idx3-ubyte",
                                  imgDir + "t10k-labels.idx1-ubyte");    // 10,000

    const int nTrain = (int)train.size();

    linAlge::mat X = makeX(train, nTrain);
    linAlge::mat Y = makeY(train, nTrain);

    // Architecture: 784 → 256 (ReLU) → 10 (softmax)
    // W1: He init (fan-in = 784, ReLU hidden)
    // W2: Xavier init (fan-in = 256, softmax output)
    std::vector<linAlge::mat> W = {
        makeWeight(784, 256, 42, /*he=*/true),   // hidden layer
        makeWeight(256,  10, 43, /*he=*/false),  // output layer
    };

    // Full forward pass for CE reporting (uses predict() = ReLU + softmax).
    auto ce = [&]() {
        linAlge::mat Xc = X;
        return model::objFunction::calcCE_cat(Xc, W, Y,
                                              model::actFunction::relu);
    };

    std::cout << "Initial CE: " << ce() << "\n";

    // Mini-batch SGD setup.
    // batch_size = 64; steps per epoch ≈ 60000/64 = 937.
    // Phase 1: ~5 epochs  (theta=0.30)
    // Phase 2: ~8 epochs  (theta=0.20)
    // Phase 3: ~7 epochs  (theta=0.10)
    // Total: ~20 epochs of mini-batch updates.
    constexpr int BS = 64;
    constexpr int SPE = 60000 / BS;   // 937 steps per epoch

    std::cout << "\n-- phase 1  theta=0.30  ~5 epochs --\n";
    optimizer::gradientDescent(
        model::objFunction::CE_cat, model::objFunction::dCE,
        model::actFunction::relu,   model::actFunction::drelu,
        W, X, Y,
        /*theta=*/0.30, /*gn=*/5 * SPE, /*printCur=*/SPE,
        /*offset=*/0, /*combined=*/true, /*lambda=*/1e-4,
        /*softmax=*/true, /*batch_size=*/BS);

    std::cout << "\n-- phase 2  theta=0.20  ~8 epochs --\n";
    optimizer::gradientDescent(
        model::objFunction::CE_cat, model::objFunction::dCE,
        model::actFunction::relu,   model::actFunction::drelu,
        W, X, Y,
        /*theta=*/0.20, /*gn=*/8 * SPE, /*printCur=*/SPE,
        /*offset=*/5 * SPE, /*combined=*/true, /*lambda=*/1e-4,
        /*softmax=*/true, /*batch_size=*/BS);

    std::cout << "\n-- phase 3  theta=0.10  ~7 epochs --\n";
    optimizer::gradientDescent(
        model::objFunction::CE_cat, model::objFunction::dCE,
        model::actFunction::relu,   model::actFunction::drelu,
        W, X, Y,
        /*theta=*/0.10, /*gn=*/7 * SPE, /*printCur=*/SPE,
        /*offset=*/13 * SPE, /*combined=*/true, /*lambda=*/1e-4,
        /*softmax=*/true, /*batch_size=*/BS);

    std::cout << "\nFinal CE (full train set): " << ce() << "\n";

    evaluate("train (60000)", train, W);
    evaluate("test  (10000)", test,  W);

    return 0;
}
