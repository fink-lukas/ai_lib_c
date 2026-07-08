#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef float (*activationFunction)(float);
typedef float (*derivative)(float);

typedef struct {
    activationFunction act;
    derivative der;
} activation;

extern const activation IDENTITY;
extern const activation BINARY_STEP;
extern const activation SIGMOID;
extern const activation TANH;
extern const activation SOFTSIGN;
extern const activation RELU;

#endif