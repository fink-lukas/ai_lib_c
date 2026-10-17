#include <math.h>
#include <string.h>
#include "activations.h"

void identity(const float* restrict inputs, float* restrict outputs, int size){
    memcpy(outputs, inputs, size * sizeof(float));
}

void d_identity(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size){
    (void)z;
    (void)a;
    memcpy(deltas, upstream_gradients, size * sizeof(float));
}

const Activation IDENTITY = {
    .act = identity,
    .der = d_identity,
    .gain = 1.0f
};



void binary_step(const float* restrict inputs, float* restrict outputs, int size) {
    for (int i = 0; i < size; i++) {
        outputs[i] = (inputs[i] >= 0.0f) ? 1.0f : 0.0f;
    }
}

void d_binary_step(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size){
    (void)a;
    (void)z;
    (void) upstream_gradients;
    memset(deltas, 0, size * sizeof(float));
}

const Activation BINARY_STEP = {
    .act = binary_step,
    .der = d_binary_step,
    .gain = 1.0f
};


void sigmoid(const float* restrict inputs, float* restrict outputs, int size){
    for(int i = 0; i < size; i++){
        outputs[i] = 1.0f/(1.0f + expf(-inputs[i]));
    }
}


void d_sigmoid(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size){
    //deltas = upstream_gradients * sigmoid(outputs) * (1.0f - sigmoid(outputs))
    (void)z;
    for(int i = 0; i < size; i++){
        deltas[i] = upstream_gradients[i] * a[i] * (1.0f - a[i]);
    }
    
}

const Activation SIGMOID = {
    .act = sigmoid,
    .der = d_sigmoid,
    .gain = 1.0f
};


void net_tanh(const float* restrict inputs, float* restrict outputs, int size){
    for(int i = 0; i < size; i++){
        outputs[i] = tanhf(inputs[i]);
    }
}

void d_net_tanh(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size){
    (void)z;
    for(int i = 0; i < size; i++){
        deltas[i] = upstream_gradients[i] * (1.0f - a[i] * a[i]);   
    }
}

const Activation TANH = {
    .act = net_tanh,
    .der = d_net_tanh,
    .gain = 1.667f //approx. 5/3
};


void softsign(const float* restrict inputs, float* restrict outputs, int size){
    for(int i = 0; i < size; i++){
        outputs[i] = inputs[i]/(1.0f + fabsf(inputs[i]));
    }
}

void d_softsign(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size) {
    (void)z;
    for (int i = 0; i < size; i++) {
        float abs_a = fabsf(a[i]);
        float term = 1.0f - abs_a;
        
        deltas[i] = upstream_gradients[i] * (term * term);
    }
}

const Activation SOFTSIGN = {
    .act = softsign,
    .der = d_softsign,
    .gain = 1.0f
};



void relu(const float* restrict inputs, float* restrict outputs, int size){
    for(int i = 0; i < size; i++){
        outputs[i] = fmaxf(0.0f, inputs[i]);
    }
}

void d_relu(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size){
    (void)a;

    for(int i = 0; i < size; i++){
        deltas[i] = (z[i] < 0.0f) ? 0.0f : upstream_gradients[i];
    }
}

const Activation RELU = {
    .act = relu,
    .der = d_relu,
    .gain = 1.414f //approx. sqrt(2)
};

void softmax(const float* restrict inputs, float* restrict outputs, int size){
    if(size <= 0) return;

    float max_val = inputs[0];
    for(int i = 0; i < size; i++){
        if(inputs[i] > max_val) max_val = inputs[i];   
    }

    float sum = 0.0f;
    for(int i = 0; i < size; i++){
        outputs[i] = expf(inputs[i] - max_val);
        sum += outputs[i];
    }

    float inv_sum = 1.0f / sum;
    for(int i = 0; i < size; i++){
        outputs[i] *= inv_sum;
    }    
    
}

void d_softmax(const float* restrict upstream_gradients, const float* restrict z, const float* restrict a, float* restrict deltas, int size){

    (void) z;
    float S = 0.0f;
    for(int i = 0; i < size; i++){
        S += a[i] * upstream_gradients[i];
    }

    for(int i = 0; i < size; i++){
        deltas[i] = a[i] * (upstream_gradients[i] - S);
    }


}

const Activation SOFTMAX = {
    .act = softmax,
    .der = d_softmax,
    .gain = 1.0f
};

//NOTE: More activations may be added later but probably not. This does the trick for what I wanna do