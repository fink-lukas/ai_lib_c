#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "activations.h"

typedef struct {
    Layer **layers;
    int layer_count;
} Network;

Network* network_create(int *dimensions, Activation *activations, int dimension_count);
void network_destroy(Network *net);
void network_forward(Network *net, float *inputs, float *outputs);
void network_backward(Network *net, float *loss_gradients, float learning_rate);

#endif