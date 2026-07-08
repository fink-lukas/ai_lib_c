#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "perceptron.h"


Layer* layer_create(int input_dim, int output_dim, activation activation_function){
    Layer* l = malloc(sizeof(Layer));
    if(l == NULL) {
        printf("ERROR: OOM");
        return NULL;
    }

    l->input_dim = input_dim;
    l->output_dim = output_dim;
    
    Perceptron** perceptrons = malloc(output_dim * sizeof(Perceptron*));
    if(perceptrons == NULL) goto perceptrons_clean_up;
    
    l->neurons = perceptrons;


    l->last_inputs = malloc(input_dim * sizeof(float));
    if(l->last_inputs == NULL) goto last_inputs_clean_up;

    l->deltas = malloc(output_dim * sizeof(float));
    if(l->deltas == NULL) goto deltas_clean_up;

    l->outputs_pre_act = malloc(output_dim * sizeof(float));
    if(l->outputs_pre_act == NULL) goto outputs_pre_act_clean_up;

    for(int i = 0; i < l->input_dim; i++){
        l->last_inputs[i] = 0.0f;
    }

    int preError = 0;
    for(int i = 0; i < output_dim; i++){
        perceptrons[i] = perceptron_create(input_dim);
        if(perceptrons[i] == NULL){
            preError = i;
            goto individual_perceptrons_clean_up;
        }
        l->deltas[i] = 0.0f;
        l->outputs_pre_act[i] = 0.0f;
    }


    l->activation_functions = activation_function;

    return l;

    individual_perceptrons_clean_up:
        for(int i = 0; i < preError; i++){
            perceptron_destroy(perceptrons[i]); 
        }

    outputs_pre_act_clean_up:
        free(l->deltas);
    deltas_clean_up:
        free(l->last_inputs);
    last_inputs_clean_up:
        free(l->neurons);
    perceptrons_clean_up:
        free(l);
        printf("ERROR OOM during Layer creation");
        return NULL;
    
}



void layer_destroy(Layer *l){
    if(l == NULL){
        return;
    }


    for(int i = 0; i < l->output_dim; i++){
        perceptron_destroy(l->neurons[i]);
    }

    free(l->deltas);
    free(l->outputs_pre_act);
    free(l->last_inputs);
    free(l->neurons);
    free(l);
}

void layer_forward(Layer *l, float *inputs, float *outputs){
    if(l == NULL || inputs == NULL || outputs == NULL) return;

    for(int i = 0; i < l->input_dim; i++){
        l->last_inputs[i] = inputs[i];
    }

    for(int output = 0; output < l->output_dim; output++){

        float percVal = perceptron_forward(l->neurons[output], inputs);
        l->outputs_pre_act[output] = percVal;
        outputs[output] = l->activation_functions.act(percVal);

    }

}
