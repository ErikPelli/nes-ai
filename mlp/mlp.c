#include <fix16.h>
#include <stdint.h>

#include "mlp.h"
#include "weights.h"

/**
 * @brief ReLU (Rectified Linear Unit) activation function.
 * @return x if x > 0, else 0.
 */
static fix16_t fix16_relu(fix16_t x) {
    return (x > 0) ? x : 0;
}

/**
 * @brief Stable Softmax activation function.
 * Uses stabilization (by subtracting maximum value) to avoid overflow within fix16_exp.
 *
 * @param input  Input array (raw logit).
 * @param output Output array (probability).
 * @param size   Size of the input & output arrays.
 */
static void stable_softmax_fix16(const fix16_t *input, fix16_t *output, uint8_t size) {
    uint8_t i;
    fix16_t max_val;
    fix16_t sum_exp;

    // Find the maximum value inside the input (to perform stable softmax)
    max_val = input[0];
    for (i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Calculate every exp(x - max(x)) and their sum
    sum_exp = 0;
    for (i = 0; i < size; ++i) {
        fix16_t exp_val = fix16_exp(fix16_sub(input[i], max_val));
        // Use output array as a temporary buffer and save there the calculated exponent
        output[i] = exp_val;
        sum_exp = fix16_add(sum_exp, exp_val);
    }

    // Divide every exponent by the sum of the exponents
    // (and avoid division by 0)
    if (sum_exp == 0) {
        sum_exp = fix16_one;
    }

    for (i = 0; i < size; ++i) {
        output[i] = fix16_div(output[i], sum_exp);
    }
}

/**
 * @brief Generic function for a Dense (Fully Connected) layer.
 * output = activation(dot(input, kernel) + bias)
 *
 * @param input          Array that contains the data to process in this layer.
 * @param input_size     Length of the input array.
 * @param output         Array where we need to save the result of this layer.
 * @param output_size    Number of neurons in this layer.
 * @param weights_cursor Cursor that iterates over the weights data and moves forward with each reading.
 * @param activation     Pointer to the activation function that we need to apply (e.g. fix16_relu).
 */
static void mlp_dense_layer(
    const fix16_t *input,
    uint8_t input_size,
    fix16_t *output,
    uint8_t output_size,
    const fix16_t **weights_cursor,
    fix16_t (*activation)(fix16_t)
) {
    uint8_t i, j;
    fix16_t sum;
    fix16_t weight;

    // The format of the weights is [bias, weight1, weight2, ...]
    for (i = 0; i < output_size; ++i) {
        // Load the bias for this neuron
        sum = *(*weights_cursor)++;

        // Calculate dot product with weights
        for (j = 0; j < input_size; ++j) {
            weight = *(*weights_cursor)++;
            // fix16_sadd avoids a incorrect values if there is an overflow
            sum = fix16_sadd(sum, fix16_mul(input[j], weight));
        }

        // Apply the activation function if it's available
        if (activation) {
            output[i] = activation(sum);
        } else {
            output[i] = sum;
        }
    }
}

// Intermediate buffers to save the results of each layer.
// A 6502 has only 256 bytes of stack space, so we must avoid saving them there.
static fix16_t _layer1_output[MLP_LAYER1_OUTPUT_SIZE];
static fix16_t _layer2_output[MLP_LAYER2_OUTPUT_SIZE];

void mlp_forward(const fix16_t *input, fix16_t *output) {
    const fix16_t *weights_cursor = WEIGHTS;

    // --- Layer 1 (Hidden): Input(49) -> Output(49), Activation: ReLU ---
    mlp_dense_layer(
        input,
        MLP_LAYER1_INPUT_SIZE,
        _layer1_output,
        MLP_LAYER1_OUTPUT_SIZE,
        &weights_cursor,
        fix16_relu
    );

    // --- Layer 2 (Hidden): Input(49) -> Output(24), Activation: ReLU ---
    mlp_dense_layer(
        _layer1_output,
        MLP_LAYER1_OUTPUT_SIZE,
        _layer2_output,
        MLP_LAYER2_OUTPUT_SIZE,
        &weights_cursor,
        fix16_relu
    );

    // --- Layer 3 (Output): Input(24) -> Output(10), Activation: Softmax ---
    mlp_dense_layer(
        _layer2_output,
        MLP_LAYER2_OUTPUT_SIZE,
        output,
        MLP_OUTPUT_SIZE,
        &weights_cursor,
        // No activation function, we call softmax later
        (void*) 0
    );
    stable_softmax_fix16(output, output, MLP_OUTPUT_SIZE);
}
