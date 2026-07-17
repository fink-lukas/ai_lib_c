#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef float (*activationFunction)(float);
typedef float (*derivative)(float);

typedef struct {
    activationFunction act;
    derivative der;
} Activation;

extern const Activation IDENTITY;
extern const Activation BINARY_STEP;
extern const Activation SIGMOID;
extern const Activation TANH;
extern const Activation SOFTSIGN;
extern const Activation RELU;

#endif