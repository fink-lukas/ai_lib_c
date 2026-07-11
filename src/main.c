#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "layer.h"
#include "activations.h"

float polynomial(float x){
    //f(x) = 0.002 x^(4)-0.01f x^(3)-0.06f x^(2)-2
    //return 0.002f * x * x * x * x - 0.01f * x * x * x -0.06f * x * x - 2.0f;

    //f(x)=0.02 x^(4)+0.01 x^(3)-0.6 x^(2)+0.3 x -4
    return 0.02f * x * x * x * x + 0.01f * x * x * x - 0.6 * x * x + 0.3f * x - 4.0f; 
}

int main() {
    
    srand(time(NULL));

    Layer *l1 = layer_create(1, 8, TANH);
    Layer *l2 = layer_create(8, 1, IDENTITY);
    float lr = 0.001f;

    float inputs[1];
    float hidden_outputs[8];
    float final_prediction[1];

    float loss_gradient[1];
    float hidden_errors[8];
    float dummy_buffer[1];

    //printf("Model initialized. Starting training (I hope lol)\n\n\n");

    printf("EPOCH;X;TARGET;PREDICTION;ERROR\n");
    for(int epoch = 0; epoch < 1000000; epoch ++){
        float x = (float)rand() / RAND_MAX * 10 - 5;
        float target = polynomial(x);
        
        //forward pass
        inputs[0] = x;
        layer_forward(l1, inputs, hidden_outputs);
        layer_forward(l2, hidden_outputs, final_prediction);

        loss_gradient[0] = final_prediction[0] - target;
        printf("%i;%f;%f;%f;%f\n", epoch, x, target, final_prediction[0], loss_gradient[0]);



        layer_backward(l2, loss_gradient, hidden_errors, lr);
        layer_backward(l1, hidden_errors, dummy_buffer, lr);
        
        
        
        
        //printf("f(%f) = %f\n", x, target);
    }

}