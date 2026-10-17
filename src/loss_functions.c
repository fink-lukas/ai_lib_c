#include <math.h>
#include <string.h>
#include "loss_functions.h"

//typedef float (*loss_forward_fn)(const float* restrict predictions, const float* restrict targets, int size);
//typedef void (*loss_backward_fn)(const float* restrict predictions, const float* restrict targets, float* restrict loss_grads, int size);


float mse_fw(const float* restrict predictions, const float* restrict targets, int size){
    float term = 1.0/(float)size;
    float sum = 0.0f;

    for(int i = 0; i < size; i++){
        sum += (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
    }
    return sum * term;
}

void mse_bw(const float* restrict predictions, const float* restrict targets, float* restrict loss_grads, int size){
    float term = 2.0/(float)size;
    for(int i = 0; i < size; i++){
        loss_grads[i] = term * (predictions[i] - targets[i]);
    }
}

const Loss MSE = {
    .forward = mse_fw,
    .backward = mse_bw
};

float mae_fw(const float* restrict predictions, const float* restrict targets, int size){
    float term = 1.0/(float)size;
    float sum = 0.0f;

    for(int i = 0; i < size; i++){
        sum += fabsf(predictions[i] - targets[i]);
    }
    return sum * term;
}

void mae_bw(const float* restrict predictions, const float* restrict targets, float* restrict loss_grads, int size){
    float term = 1.0/(float)size;

    for(int i = 0; i < size; i++){
        float sgn = (float)(predictions[i] > targets[i]) - (float)(predictions[i] < targets[i]);
        loss_grads[i] = term * sgn;
    }
}

const Loss MAE = {
    .forward = mae_fw,
    .backward = mae_bw
};


float bce_fw(const float* restrict predictions, const float* restrict targets, int size){
    float term = - 1.0 / (float) size;
    float sum = 0; 
    for(int i = 0; i < size; i++){
        sum += targets[i] * logf(predictions[i]) + (1 - targets[i]) * logf(1 - predictions[i]);
    }

    return term * sum;
}

void bce_bw(const float* restrict predictions, const float* restrict targets, float* restrict loss_grads, int size){
    float term = 1.0 / (float)size;
    for(int i = 0; i < size; i++){
        loss_grads[i] = term * ((predictions[i] - targets[i])/(predictions[i] * (1 - predictions[i])));
    }
}

const Loss BCE = {
    .forward = bce_fw,
    .backward = bce_bw
};


float cce_fw(const float* restrict predictions, const float* restrict targets, int size){
    int k = 0;

    for(int i = 0; i < size; i++){
        if(targets[i] == 1){
            k = i; 
            break;
        }
    }

    return -( targets[k] * logf(fmaxf(predictions[k], 1e-7f)));
}

void cce_bw(const float* restrict predictions, const float* restrict targets, float* restrict loss_grads, int size){
    int k = 0;
    memset(loss_grads, 0, size * sizeof(float));
    
    for(int i = 0; i < size; i++){
        if(targets[i] == 1){
            k = i;
            break;
        }
    }

    loss_grads[k] = -1.0f/fmaxf(predictions[k], 1e-7f);

}

const Loss CCE = {
    .backward = cce_bw,
    .forward = cce_fw
};

