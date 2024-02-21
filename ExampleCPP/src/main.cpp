#include <math.h>
#include <stdio.h>

#include <backward/backward.hpp>

#include "Layer.h"
#include "Trainer.h"


int main(int argc, char *argv[]) {
  backward::SignalHandling sh;

  printf("Neural Network Example in C++\n");

  // Training data to learn XOR function
  std::vector<std::vector<double>> trainingInputs = {
      {0.0f, 0.0f}, 
      {1.0f, 0.0f}, 
      {0.0f, 1.0f}, 
      {1.0f, 1.0f}};
  std::vector<std::vector<double>> trainingOutputs = {
      {0.0f}, 
      {1.0f}, 
      {0.0f}, 
      {1.0f}};


  const int hiddenNodeCount = 2;

  // Create network layers
  auto inputLayer = NeuralNetwork::Layer::Create(trainingInputs[0].size());
  auto hiddenLayer = NeuralNetwork::Layer::Create(hiddenNodeCount);
  auto outputLayer = NeuralNetwork::Layer::Create(trainingOutputs[0].size());

  // Connect layers
  outputLayer->setInput(hiddenLayer);
  hiddenLayer->setInput(inputLayer);

  // Training loop
  int numberOfEpochs = 10000;

  NeuralNetwork::Trainer trainer(inputLayer, trainingInputs, trainingOutputs);

  trainer.train(numberOfEpochs);

  return EXIT_SUCCESS;
}