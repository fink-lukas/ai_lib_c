#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "layer.h"


Layer* layer_create(int input_dim, int output_dim, Activation activation_function){
    Layer* l = calloc(1, sizeof(Layer));
    if(l == NULL) {
        printf("ERROR: OOM");
        return NULL;
    }

    l->input_dim = input_dim;
    l->output_dim = output_dim;
    
    l->weights = malloc(output_dim * input_dim * sizeof(float));

    l->biases = calloc(output_dim, sizeof(float));

    l->last_inputs = calloc(input_dim, sizeof(float));

    l->deltas = calloc(output_dim, sizeof(float));

    l->outputs_pre_act = calloc(output_dim, sizeof(float));

    l->outputs_post_act = calloc(output_dim, sizeof(float));

    float scale = activation_function.gain * sqrtf(3.0f / (float)input_dim);

    for(int i = 0; i < output_dim * input_dim; i++){
        float random_value = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
            l->weights[i] = random_value * scale;
    }


    l->activation_functions = activation_function;

    if (!l->weights || !l->biases || !l->last_inputs || !l->deltas || !l->outputs_pre_act || !l->outputs_post_act){
        fprintf(stderr, "ERROR: OOM during Layer creation\n");
        layer_destroy(l);
        return NULL;
    }

    return l;
    
}



void layer_destroy(Layer *l){
    if(l == NULL) return;
    
    free(l->deltas);
    free(l->outputs_pre_act);
    free(l->outputs_post_act);
    free(l->last_inputs);
    free(l->weights);
    free(l->biases);
    free(l);
}

void layer_forward(Layer *l, float *inputs, float *outputs){
    if(l == NULL || inputs == NULL || outputs == NULL) return;

    if(l->last_inputs != inputs){
        memcpy(l->last_inputs, inputs, l->input_dim * sizeof(float));
    }

    for(int output = 0; output < l->output_dim; output++){

        int offset = output * l->input_dim;
        l->outputs_pre_act[output] = l->biases[output];
        for(int weight = 0; weight < l->input_dim; weight++) {
            l->outputs_pre_act[output] += l->weights[offset + weight] * inputs[weight];
        }

    }

    l->activation_functions.act(l->outputs_pre_act, outputs, l->output_dim);
    memcpy(l->outputs_post_act, outputs, l->output_dim * sizeof(float));

}

void layer_backward(Layer *l, float *upstream_gradients, float *downstream_gradients, float learning_rate){
    if (l == NULL || upstream_gradients == NULL || downstream_gradients == NULL) return;

    l->activation_functions.der(upstream_gradients, l->outputs_pre_act, l->outputs_post_act, l->deltas, l->output_dim);

    for(int i = 0; i < l->input_dim; i++){
        downstream_gradients[i] = 0.0f;
    }

    for(int j = 0; j < l->output_dim; j++){
        l->biases[j] -= learning_rate * l->deltas[j];

        int offset = j * l->input_dim;
        for(int weight = 0; weight < l->input_dim; weight++){
            downstream_gradients[weight] += l->weights[offset + weight] * l->deltas[j];
            l->weights[offset + weight] -= learning_rate * l->deltas[j] * l->last_inputs[weight];
        }

    }

}