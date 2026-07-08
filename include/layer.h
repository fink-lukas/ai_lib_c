#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"
#include "activations.h"

typedef struct {
    Perceptron **neurons;
    int input_dim;
    int output_dim;


    float *last_inputs;
    float *outputs_pre_act;
    float *deltas;

    //pointer to activation function
    activation activation_functions;
} Layer;

Layer* layer_create(int input_dim, int output_dim, activation);
void layer_destroy(Layer *l);

void layer_forward(Layer *l, float *inputs, float *outputs);



#endif