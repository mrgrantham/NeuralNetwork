#include "Layer.h"

#include <fmt/format.h>
#include <iostream>
#include <memory>

namespace NeuralNetwork {

std::shared_ptr<Layer> Layer::Create(size_t nodeCount) {
  return std::shared_ptr<Layer>(new Layer(nodeCount));
}

std::shared_ptr<Layer> Layer::getFirstLayer() {
  auto layer = shared_from_this();
  while (layer->inputLayer_) {
    layer = layer->inputLayer_;
  }
  return layer;
}

std::shared_ptr<Layer> Layer::getFinalLayer() {
  auto layer = shared_from_this();
  while (layer->outputLayer_) {
    layer = layer->outputLayer_;
  }
  return layer;
}

void Layer::setInput(std::shared_ptr<Layer> layer) {
  if (!layer) {
    std::cout << "Input layer is null, will assume this is first layer"
              << std::endl;
    return;
  }

  // Initialize weight array connecting input layer to this layer
  inputLayer_ = layer;
  std::cout << "setting output" << std::endl;
  inputLayer_->setOutput(shared_from_this());
  auto inputNodeCount = inputLayer_->nodeCount();
  weights_.reserve(inputNodeCount);
  // Populate initial random weight
  for (int inputNodeIndex = 0; inputNodeIndex < inputNodeCount;
       inputNodeIndex++) {

    weights_.emplace_back(std::vector<double>(this->nodeCount()));
    for (int nodeIndex = 0; nodeIndex < this->nodeCount(); nodeIndex++) {
      // Weighted connections between a node from the input
      // layer and from this layer
      fmt::print("initializing weight for inputNodeIndex:{}  nodeIndex:{}\n",
                 inputNodeIndex, nodeIndex);
      weights_[inputNodeIndex][nodeIndex] = initializeWeight();
    }
  }
}

std::vector<double> Layer::predict(const std::vector<double> &inputs) {
  // If no input layer is set then this is the input layer so we just
  // assessing inputs directly
  if (!inputLayer_) {
    values_ = inputs;
  } else {
    // If there is an input layer then calculate values_ using the weights,
    // biases and the values_ of the input layer
    for (int nodeIndex = 0; nodeIndex < this->nodeCount(); nodeIndex++) {
      double activation = biases_[nodeIndex];

      for (int inputLayerNodeIndex = 0; inputLayerNodeIndex < inputs.size();
           inputLayerNodeIndex++) {
        activation += inputs[inputLayerNodeIndex] *
                      weights_[inputLayerNodeIndex][nodeIndex];
      }
      values_[nodeIndex] = sigmoid(activation);
    }
  }

  // If there is n output layer set then we still need to get the results from
  // that layer otherwise we just
  if (outputLayer_) {
    return outputLayer_->predict(values_);
  } else {
    return values_;
  }
}

int Layer::nodeCount() const { return nodeCount_; }

void Layer::printWeights() const {
  printf("Weights\n");
  for (int nodeIndex = 0; nodeIndex < this->nodeCount(); nodeIndex++) {
    printf("[ ");
    for (int inputNodeIndex = 0;
         inputNodeIndex < this->inputLayer_->nodeCount(); inputNodeIndex++) {
      printf("%f ", this->weights_[inputNodeIndex][nodeIndex]);
    }
    printf(" ]");
  }
  printf("\n\n");
}

void Layer::printBiases() const {
  printf("Biases\n");
  for (int nodeIndex = 0; nodeIndex < this->nodeCount(); nodeIndex++) {
    printf("%f ", this->biases_[nodeIndex]);
  }
  printf("\n\n");
}

Layer::Layer(size_t nodeCount) : nodeCount_(nodeCount) {
  biases_.reserve(nodeCount);
  values_.reserve(nodeCount);
  for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
    printf("initializing bias for nodeIndex:%d\n", nodeIndex);
    biases_.push_back(initializeWeight());
    values_.push_back(0.0f);
  }
}
void Layer::setOutput(std::shared_ptr<Layer> layer) { outputLayer_ = layer; }

// Activation function
double Layer::sigmoid(double x) { return 1 / (1 + exp(-x)); }

double Layer::initializeWeight() {
  auto weight = (double)rand() / (double)RAND_MAX;
  printf("initialized weight: %f\n", weight);
  return weight;
}

} // namespace NeuralNetwork