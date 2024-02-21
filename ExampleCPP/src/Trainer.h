#pragma once
#include "Layer.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

namespace NeuralNetwork {
class Trainer {
private:
  const std::vector<std::vector<double>> &trainingInput_;
  const std::vector<std::vector<double>> &trainingOutput_;
  std::vector<size_t> trainingOrder_;
  std::shared_ptr<Layer> model_;
  double learningRate_;

  // Setup random devices for use when shuffling training data between epochs
  std::random_device randomDevice_;
  std::mt19937 generator_;

  template <typename Collection> void shuffle(Collection &trainingOrder) {
    // Shuffle that list of indexes to allow for random training. This is less
    // expensive than shuffling the actual training data
    std::shuffle(trainingOrder.begin(), trainingOrder.end(), generator_);
  }

public:
  Trainer(std::shared_ptr<Layer> model,
          const std::vector<std::vector<double>> &trainingInput,
          const std::vector<std::vector<double>> &trainingOutput);

  // Derivative of sigmoid activation function in Layer.h
  // TODO: Make activation function and derivative something that you pass into
  // the model/layer

  double dSigmoid(double x);

  void train(int epochs);

  // TODO: Implement
  void validate();
};
} // namespace NeuralNetwork