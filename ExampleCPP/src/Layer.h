#pragma once

#include <fmt/format.h>
#include <iostream>
#include <memory>

namespace NeuralNetwork {

class Layer : public std::enable_shared_from_this<Layer> {
public:
  friend class Trainer;
  static std::shared_ptr<Layer> Create(size_t nodeCount);

  std::shared_ptr<Layer> getFirstLayer();

  std::shared_ptr<Layer> getFinalLayer();

  void setInput(std::shared_ptr<Layer> layer);

  std::vector<double> predict(const std::vector<double> &inputs);

  int nodeCount() const;

  void printWeights() const;

  void printBiases() const;

private:
  Layer(size_t nodeCount);
  void setOutput(std::shared_ptr<Layer> layer);
  // Activation function
  double sigmoid(double x);

  double initializeWeight();

  const int nodeCount_;
  std::vector<double> biases_;
  std::shared_ptr<Layer> inputLayer_;
  std::shared_ptr<Layer> outputLayer_;
  // weights for connections between input layer and this layer
  std::vector<std::vector<double>> weights_;

  // Calculated values for the current prediction run
  // For the input layer this will just contain the inputs
  // This gets overwritten with every call to predict()
  std::vector<double> values_;
};

} // namespace NeuralNetwork