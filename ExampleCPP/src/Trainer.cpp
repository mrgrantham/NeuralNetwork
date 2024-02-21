

#include "Trainer.h"
#include "Layer.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

namespace NeuralNetwork {
Trainer::Trainer(std::shared_ptr<Layer> model,
                 const std::vector<std::vector<double>> &trainingInput,
                 const std::vector<std::vector<double>> &trainingOutput)
    : trainingInput_(trainingInput), trainingOutput_(trainingOutput),
      trainingOrder_(trainingInput_.size()), model_(model), learningRate_(0.1f),
      randomDevice_(), generator_(randomDevice_()) {
  assert(trainingInput_.size() == trainingOutput_.size());
  // Create a list representing the indexes of all the training input vectors
  std::iota(std::begin(trainingOrder_), std::end(trainingOrder_), 0);
}

// Derivative of sigmoid activation function in Layer.h
// TODO: Make activation function and derivative something that you pass into
// the model/layer

double Trainer::dSigmoid(double x) { return x * (1 - x); }

void Trainer::train(int epochs) {
  // Train the neural network for a number of epochs
  for (int epoch = 0; epoch < epochs; epoch++) {
    printf("\n======  STARTING EPOCH %d ======\n", epoch);
    shuffle(trainingOrder_);
    std::cout << "TRAINING ORDER [ ";
    for (auto item : trainingOrder_) {
      std::cout << item << " ";
    }
    std::cout << "]" << std::endl;

    for (int trainingOrderIndex = 0; trainingOrderIndex < trainingInput_.size();
         trainingOrderIndex++) {

      int trainingSetIndex = trainingOrder_[trainingOrderIndex];

      auto &currentTrainingInput = trainingInput_[trainingSetIndex];
      // auto &currentTrainingOutput = trainingOutput_[trainingSetIndex];

      // Forward pass, also just called regular prediction with
      // the current model
      auto outputResults = model_->predict(currentTrainingInput);

      printf("Input: %g, %g Output: %g Predicted Output: %g \n",
             trainingInput_[trainingSetIndex][0],
             trainingInput_[trainingSetIndex][1], outputResults[0],
             trainingOutput_[trainingSetIndex][0]);

      // Backpropagation pass

      // Get the final output layer
      const auto finalLayer = model_->getFinalLayer();

      auto currentLayer = finalLayer;

      std::vector<std::vector<double>> weightDeltas;

      // Compute deltas in output weights
      while (currentLayer->inputLayer_) {
        // create empty vector for this layers weight deltas
        weightDeltas.emplace_back(currentLayer->nodeCount());
        for (int nodeIndex = 0; nodeIndex < currentLayer->nodeCount();
             nodeIndex++) {

          double error = 0.0f;
          if (currentLayer->outputLayer_) {
            auto outputLayer = currentLayer->outputLayer_;
            // current layer has output layer
            // use delta from output layer to compute error in this layer
            for (int outputLayerIndex = 0;
                 outputLayerIndex < outputLayer->nodeCount();
                 outputLayerIndex++) {
              // get weightDeltas for previous layer
              auto &outputLayerWeights = *(weightDeltas.rbegin() + 1);
              error += outputLayerWeights[outputLayerIndex] *
                       outputLayer->weights_[nodeIndex][outputLayerIndex];
            }
          } else {
            // current layer is output layer
            // Difference between expected output and actual
            error = (trainingOutput_[trainingSetIndex][nodeIndex] -
                     currentLayer->values_[nodeIndex]);
          }

          // calculcate delta from expected/correct output with error
          weightDeltas.back()[nodeIndex] =
              error * dSigmoid(currentLayer->values_[nodeIndex]);
        }

        // Cycle to next layer
        currentLayer = currentLayer->inputLayer_;
      }

      // Apply changes in the output weights using the learning rate
      // reset the working currentLayer to the final layer

      currentLayer = finalLayer;
      int layerDeltaIndex =
          0; // index for accessing the deltas associated with currentLayer
      while (currentLayer->inputLayer_) {

        for (int nodeIndex = 0; nodeIndex < currentLayer->nodeCount();
             nodeIndex++) {
          currentLayer->biases_[nodeIndex] +=
              weightDeltas[layerDeltaIndex][nodeIndex] * learningRate_;
          auto inputLayer = currentLayer->inputLayer_;
          for (int inputNodeIndex = 0; inputNodeIndex < inputLayer->nodeCount();
               inputNodeIndex++) {
            currentLayer->weights_[inputNodeIndex][nodeIndex] +=
                inputLayer->values_[inputNodeIndex] *
                weightDeltas[layerDeltaIndex][nodeIndex] * learningRate_;
          }
        }

        // Cycle to next layer
        currentLayer = currentLayer->inputLayer_;
        // update layerDeltaIndex to match
        layerDeltaIndex++;
      }

      currentLayer = finalLayer;
      while (currentLayer->inputLayer_) {
        // Print final weights/biases after training is done for this epoch

        // currentLayer->printWeights();
        // currentLayer->printBiases();

        // Cycle to next layer
        currentLayer = currentLayer->inputLayer_;
      }
    }
  }
}

void Trainer::validate() {}
} // namespace NeuralNetwork