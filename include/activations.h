#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef void (*activationFunction)(const float* restrict inputs, float* restrict outputs, int size);
typedef void (*back)(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size);

typedef struct {
    activationFunction act;
    back der;
    float gain;
} Activation;

extern const Activation IDENTITY;
extern const Activation BINARY_STEP;
extern const Activation SIGMOID;
extern const Activation TANH;
extern const Activation SOFTSIGN;
extern const Activation RELU;

extern const Activation SOFTMAX;

#endif