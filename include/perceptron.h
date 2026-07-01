#ifndef PERCEPTRON_H
#define PERCEPTRON_H

typedef struct {
    float *weights;
    float bias;
    int input_dim;
} Perceptron;

Perceptron* perceptron_create(int input_dim);
void perceptron_destroy(Perceptron *p);

float perceptron_forward(Perceptron *p, float *inputs);

#endif