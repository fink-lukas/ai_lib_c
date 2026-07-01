#include <math.h>

float identity(float x){
    return x;
}

float d_identity(float x){
    return 1.0f;
}

float binary_step(float x){
    if( x < 0) return 0;
    return 1.0f;
}

float d_binary_step(float x){
    return 0;
}

float sigmoid(float x){
    return 1.0f/(1.0f + expf(-x));
}

float d_sigmoid(float x){
    return sigmoid(x)*(1.0f - sigmoid(x));
}

float net_tanh(float x){
    return (expf(x) - expf(-x))/(expf(x) + expf(-x));
}

float d_net_tanh(float x){
    return 1.0f - tanh(x) * tanh(x);
}

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

float ReLu(float x){
    if(x < 0) return 0;
    return x;
}

float d_ReLu(float x){
    if(x < 0) return 0;
    return 1.0f;
}

//NOTE: More activations may be added later but probably not. This does the trick for what I wanna do