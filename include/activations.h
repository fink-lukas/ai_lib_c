#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef void (*activationFunction)(float* inputs, float* outputs, int size);
typedef void (*back)(float* upstream_gradients, float* z, float* a, float* deltas, int size);

typedef struct {
    activationFunction act;
    back der;
} Activation;

extern const Activation IDENTITY;
extern const Activation BINARY_STEP;
extern const Activation SIGMOID;
extern const Activation TANH;
extern const Activation SOFTSIGN;
extern const Activation RELU;

extern const Activation SOFTMAX;

#endif