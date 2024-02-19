#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define inputCount 2
#define hiddenNodeCount 2
#define outputCount 1
#define trainingSetCount 4

// Activation function
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
// Derivative of sigmoid
double dSigmoid(double x) { return x * (1 - x); }

double initializeWeight() { return (double)rand() / (double)RAND_MAX; }

void shuffle(int *array, size_t size) {
  if (size > 1) {
    for (size_t index = 0; index < size - 1; index++) {
      size_t randomShuffleIndex = index + rand() / (RAND_MAX / (size - index));
      int temp = array[randomShuffleIndex];
      array[randomShuffleIndex] = array[index];
      array[index] = temp;
    }
  }
}

int main(int argc, char *argv[]) {
  printf("Neural Network Example in C\n");
  const double learningRate = 0.1f;

  // Create arrays with input, hidden, and output layers
  double hiddenLayer[hiddenNodeCount];
  double outputLayer[outputCount];

  double hiddenLayerBias[hiddenNodeCount] = {0};
  double outputLayerBias[outputCount] = {0};

  double hiddenWeights[inputCount][hiddenNodeCount] = {0};
  double outputWeights[hiddenNodeCount][outputCount] = {0};


  // Training data to learn XOR function
  double trainingInputs[trainingSetCount][inputCount] = {
      {0.0f, 0.0f}, 
      {1.0f, 0.0f}, 
      {0.0f, 1.0f}, 
      {1.0f, 1.0f}};
  double trainingOutputs[trainingSetCount][outputCount] = {
      {0.0f}, 
      {1.0f}, 
      {0.0f}, 
      {1.0f}};

  // Populate initial random weight
  for (int i = 0; i < inputCount; i++) {
    for (int j = 0; j < hiddenNodeCount; j++) {
      hiddenWeights[i][j] = initializeWeight();
    }
  }

  for (int i = 0; i < hiddenNodeCount; i++) {
    for (int j = 0; j < outputCount; j++) {
      outputWeights[i][j] = initializeWeight();
    }
  }

  // Populate initial random biases

    for (int i = 0; i < hiddenNodeCount; i++) {
      hiddenLayerBias[i] = initializeWeight();
    }

    for (int i = 0; i < outputCount; i++) {
      outputLayerBias[i] = initializeWeight();
    }

  // Forward pass

  int trainingSetOrder[] = {0,1,2,3};

  // Training loop
  int numberOfEpochs = 2000;

  // Train the neural network for a number of epochs
  for (int epoch = 0; epoch< numberOfEpochs; epoch++) {
    printf("\n======  STARTING EPOCH %d ======\n",epoch);
    shuffle(trainingSetOrder,trainingSetCount);

    for (int x = 0; x < trainingSetCount; x++) {
      int trainingSetIndex = trainingSetOrder[x];

      // Forward pass

      // Compute the hidden layer activation
      for (int hiddenNodeIndex = 0; hiddenNodeIndex < hiddenNodeCount;
           hiddenNodeIndex++) {
        double activation = hiddenLayerBias[hiddenNodeIndex];

        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
          activation += trainingInputs[trainingSetIndex][inputIndex] *
                        hiddenWeights[inputIndex][hiddenNodeIndex];
        }
        hiddenLayer[hiddenNodeIndex] = sigmoid(activation);
      }

      // Compute the output layer activation
      for (int outputNodeIndex = 0; outputNodeIndex < outputCount;
           outputNodeIndex++) {
        double activation = hiddenLayerBias[outputNodeIndex];

        for (int hiddenLayerIndex = 0; hiddenLayerIndex < hiddenNodeCount; hiddenLayerIndex++) {
          activation += hiddenLayer[hiddenLayerIndex] *
                        outputWeights[hiddenLayerIndex][outputNodeIndex];
        }
        outputLayer[outputNodeIndex] = sigmoid(activation);
      }

      printf("Input: %g, %g Output: %g Predicted Output: %g \n",
             trainingInputs[trainingSetIndex][0],
             trainingInputs[trainingSetIndex][1], outputLayer[0],
             trainingOutputs[trainingSetIndex][0]);

      // Backpropagation pass

      // Compute change in output weights
      double deltaOutput[outputCount];

      for (int j = 0; j < outputCount; j++) {
        // Difference between expected output and actual
        double error = (trainingOutputs[trainingSetIndex][j] - outputLayer[j]);

        // calaulcate delta from expected/correct output with error
        deltaOutput[j] = error * dSigmoid(outputLayer[j]);
      }

      double deltaHidden[hiddenNodeCount];
      for (int j = 0; j < hiddenNodeCount; j++) {
        double error = 0.0f;
        for (int k = 0; k < outputCount; k++) {
          error += deltaOutput[k] * outputWeights[j][k];
        }
        deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
      }

      // Apply change in the output weights
      for (int j = 0; j < outputCount; j++)
      {
        outputLayerBias[j] += deltaOutput[j] * learningRate;
        for (int k = 0 ; k < hiddenNodeCount; k++) {
          outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
        }
      }
      
      // Apply change in hidden weights
      for (int j = 0; j < hiddenNodeCount; j++)
      {
        hiddenLayerBias[j] += deltaHidden[j] * learningRate;
        for (int k = 0 ; k < inputCount; k++) {
          hiddenWeights[k][j] += trainingInputs[trainingSetIndex][k] * deltaHidden[j] * learningRate;
        }
      }

      // Print final weights/biases after training is done for this epoch
      printf("Finale Hidden Weights\n");
      for (int j = 0; j < hiddenNodeCount; j++) {
        printf("[ ");
        for (int k = 0; k < inputCount; k++) {
          printf("%f ", hiddenWeights[k][j]);
        }
        printf(" ]");
      }
      printf("\n\n");

      printf("Finale Hidden Biases\n");
      for (int j = 0; j < hiddenNodeCount; j++) {
        printf("%f ", hiddenLayerBias[j]);
      }
      printf("\n\n");

      printf("Finale Output Weights\n");
      for (int j = 0; j < outputCount; j++) {
        printf("[ ");
        for (int k = 0; k < hiddenNodeCount; k++) {
          printf("%f ", outputWeights[k][j]);
        }
        printf("] ");
      }
      printf("\n\n");

      printf("Finale Output Biases\n");
      for (int j = 0; j < outputCount; j++) {
        printf("%f ", outputLayerBias[j]);
      }
      printf("\n\n");
    }
  }

  return EXIT_SUCCESS;
}