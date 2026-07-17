#include <stdio.h>
#include <stdlib.h>

#include "network.h"
#include "layer.h"

Network* network_create(int *dimensions, Activation *activations, int dimension_count){

    if (dimensions == NULL || activations == NULL || dimension_count < 2) {
        return NULL;
    }


    Network *net = malloc(sizeof(Network));
    if(net == NULL) goto net_cleanup;


    net->layer_count = dimension_count - 1;

    int layers_initialized = 0;

    net->layers = malloc(sizeof(Layer*) * net->layer_count);
    if(net->layers == NULL) goto net_struct_cleanup;
    
    for(int i = 0; i < dimension_count - 1; i++){
        net->layers[i] = layer_create(dimensions[i], dimensions[i + 1], activations[i]);
        if(net->layers[i] == NULL) goto layers_cleanup;
        layers_initialized++;
    }

    return net;

    layers_cleanup:
        for(int i = 0; i < layers_initialized; i++){
            layer_destroy(net->layers[i]);
        }
        free(net->layers);

    net_struct_cleanup:
        free(net);


    net_cleanup:
        printf("ERROR: OOM\n");
        return NULL;

}


void network_destroy(Network *net){
    if(net == NULL) return;

    for(int i = 0; i < net->layer_count; i++){
        layer_destroy(net->layers[i]);
    }

    free(net->layers);
    free(net);

}


void network_forward(Network *net, float *inputs, float *outputs){
    float *current_inputs = inputs;

    for(int i = 0; i < net->layer_count; i++){
        float *current_outputs;

        if(i == net->layer_count - 1){
            current_outputs = outputs;
        } else{
            current_outputs = net->layers[i + 1]->last_inputs;
        }
        
        layer_forward(net->layers[i], current_inputs, current_outputs);

        current_inputs = current_outputs;
    
    }

}


void network_backward(Network *net, float *loss_gradients, float learning_rate){
    if(net == NULL || loss_gradients == NULL) return;

    int max_grad_size = 0;
    for(int i = 0; i < net->layer_count; i++){
        if(net->layers[i]->input_dim > max_grad_size){
            max_grad_size = net->layers[i]->input_dim;
        }
    }

    float buffer_a[max_grad_size];
    float buffer_b[max_grad_size];

    float *current_upstream = loss_gradients;
    float *current_downstream = buffer_a; 


    for(int i = net->layer_count - 1; i >= 0; i--){
        layer_backward(net->layers[i], current_upstream, current_downstream, learning_rate);

        current_upstream = current_downstream;
        current_downstream = (current_upstream == buffer_a) ? buffer_b : buffer_a;
    }

}