#ifndef LAYER_H
#define LAYER_H

#include "activations.h"

typedef struct {
    float *weights;
    float *biases;
    int input_dim;
    int output_dim;


    float *last_inputs;
    float *outputs_pre_act;
    float *outputs_post_act;
    float *deltas;

    //pointer to activation function
    Activation activation_functions;
} Layer;

Layer* layer_create(int input_dim, int output_dim, Activation);
void layer_destroy(Layer *l);

void layer_forward(Layer *l, float *inputs, float *outputs);
void layer_backward(Layer *l, float *upstream_gradients, float *downstream_gradients, float learning_rate);

#endif