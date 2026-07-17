#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include "layer.h"
#include "activations.h"
#include "network.h"

float polynomial(float x){
    //f(x) = 0.002 x^(4)-0.01f x^(3)-0.06f x^(2)-2
    //return 0.002f * x * x * x * x - 0.01f * x * x * x -0.06f * x * x - 2.0f;

    //f(x)=0.02 x^(4)+0.01 x^(3)-0.6 x^(2)+0.3 x -4
    return 0.02f * x * x * x * x + 0.01f * x * x * x - 0.6 * x * x + 0.3f * x - 4.0f; 
}

int main() {
    
    srand(time(NULL));

    int dims[] = {1, 8, 1};
    Activation acts[] = {TANH, IDENTITY};    

    Network *network = network_create(dims, acts, 3);
    float lr = 0.001f;
    float inputs[1];
    float final_prediction[1];
    float loss_gradient[1];



    printf("EPOCH;X;TARGET;PREDICTION;ERROR\n");
    for(int epoch = 0; epoch < 1000000; epoch ++){
        float x = (float)rand() / RAND_MAX * 10 - 5;
        float target = polynomial(x);
        inputs[0] = x;

        network_forward(network, inputs, final_prediction);

        loss_gradient[0] = final_prediction[0] - target;
        printf("%i;%f;%f;%f;%f\n", epoch, x, target, final_prediction[0], loss_gradient[0]);

        network_backward(network, loss_gradient, lr);
    }

    

}