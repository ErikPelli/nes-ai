#ifndef NES_AI_MLP_H
#define NES_AI_MLP_H

#include <fix16.h>

/**
 * @brief Performs an inference (forward pass) through the neural network.
 * Takes an input array, performs all layer calculations
 * (ReLU for hidden layers, Softmax for output) and
 * writes the results to the output array.
 *
 * @param input array of fix16_t values of length MLP_INPUT_SIZE.
 * @param output array of fix16_t values of length MLP_OUTPUT_SIZE, where we write the SoftMax probabilities.
 */
void mlp_forward(const fix16_t *input, fix16_t *output);

#endif // NES_AI_MLP_H
