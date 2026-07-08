#include <math.h>
#include "activations.h"

float identity(float x){
    return x;
}

float d_identity(float x){
    return 1.0f;
}

const activation IDENTITY = {
    .act = identity,
    .der = d_identity
};



float binary_step(float x){
    if( x < 0) return 0;
    return 1.0f;
}

float d_binary_step(float x){
    return 0;
}

const activation BINARY_STEP = {
    .act = binary_step,
    .der = d_binary_step
};




float sigmoid(float x){
    return 1.0f/(1.0f + expf(-x));
}

float d_sigmoid(float x){
    float sig = sigmoid(x);
    return sig*(1.0f - sig);
}

const activation SIGMOID = {
    .act = sigmoid,
    .der = d_sigmoid
};




float net_tanh(float x){
    return (expf(x) - expf(-x))/(expf(x) + expf(-x));
}

float d_net_tanh(float x){
    return 1.0f - net_tanh(x) * net_tanh(x);
}


const activation TANH = {
    .act = net_tanh,
    .der = d_net_tanh
};



//NOTE: smth(x) isn't included because I can't be bothered to do something smart about the function signature

float softsign(float x){

    float absx = x;
    if(x < 0) absx = -x;

    return x/(1.0f + absx);
}

float d_softsign(float x){

    float absx = x;
    if(x < 0) absx = -x;

    return 1.0f / ((1.0f + absx) * (1.0f + absx));
}

const activation SOFTSIGN = {
    .act = softsign,
    .der = d_softsign
};





float ReLu(float x){
    if(x < 0) return 0;
    return x;
}

float d_ReLu(float x){
    if(x < 0) return 0;
    return 1.0f;
}

const activation RELU = {
    .act = ReLu,
    .der = d_ReLu
};

//NOTE: More activations may be added later but probably not. This does the trick for what I wanna do