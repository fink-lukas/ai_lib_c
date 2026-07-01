#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "perceptron.h"


float step_function(float total_sum) {
    return (total_sum >= 0.0f) ? 1.0f : 0.0f;
}

int main() {
    srand(time(NULL));

    float inputs[4][2] = {
        {0.0f, 0.0f}, // 0 AND 0 = 0
        {0.0f, 1.0f}, // 0 AND 1 = 0
        {1.0f, 0.0f}, // 1 AND 0 = 0
        {1.0f, 1.0f}  // 1 AND 1 = 1
    };
    float targets[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    Perceptron *p = perceptron_create(2);
    if (p == NULL) {
        printf("Failed to allocate perceptron!\n");
        return 1;
    }

    float learning_rate = 0.1f;
    int epochs = 50; 

    printf("--- STARTING TRAINING ---\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        int total_errors = 0;

        for (int i = 0; i < 4; i++) {
            float raw_sum = perceptron_forward(p, inputs[i]);
            
            float prediction = step_function(raw_sum);
            
            float error = targets[i] - prediction;

            if (error != 0.0f) {
                total_errors++;


                for(int j = 0; j < p->input_dim; j++){
                    p->weights[j] += learning_rate * error * inputs[i][j];
                }

                p->bias += learning_rate * error;

            }
        }

        if (total_errors == 0) {
            printf("Converged early at epoch %d!\n", epoch + 1);
            break;
        }
    }

    printf("\n--- FINAL TRUTH TABLE VERIFICATION ---\n");
    for (int i = 0; i < 4; i++) {
        float raw_sum = perceptron_forward(p, inputs[i]);
        float prediction = step_function(raw_sum);
        printf("In: [%.0f, %.0f] | Target: %.0f | Predicted: %.0f\n", 
               inputs[i][0], inputs[i][1], targets[i], prediction);
    }

    perceptron_destroy(p);
    p = NULL;

    return 0;
}