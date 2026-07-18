#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "perceptron.h"

Perceptron* perceptron_create(int input_dim) {

    Perceptron *p = malloc(sizeof(Perceptron));

    if (p == NULL) {
        printf("ERROR: OOM");
        return NULL;
    }

    
    p->weights = malloc(input_dim * sizeof(float));

    if (p->weights == NULL) {
        free(p);
        printf("ERROR: OOM");
        return NULL; 
    }
    

    p->input_dim = input_dim;
    p->bias = 0.0f;

    float scale = sqrtf(2.0f / (float)input_dim);
    
    for(int i = 0; i < input_dim; i++){
        float random_value = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p->weights[i] = random_value * scale;
    }

    //TODO: Add mini batch support for computational advantages

    return p; 
}

void perceptron_destroy(Perceptron *p) {
    if (p != NULL) {
        
        free(p->weights);
        free(p);

    }
}

// 3. THE BEHAVIOR: The Math Execution (Forward Pass)
float perceptron_forward(Perceptron *p, float *inputs) {
    float total_sum = 0.0f;
 
    for(int i = 0; i < p->input_dim; i++) {
        total_sum = total_sum + inputs[i] * p->weights[i];
    }
    total_sum += p->bias;



    return total_sum;
}