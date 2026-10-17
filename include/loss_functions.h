#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H


typedef float (*loss_forward_fn)(const float* restrict predictions, const float* restrict targets, int size);
typedef void (*loss_backward_fn)(const float* restrict predictions, const float* restrict targets, float* restrict loss_grads, int size);

typedef struct {
    loss_forward_fn forward;
    loss_backward_fn backward;
} Loss;

extern const Loss MSE;
extern const Loss MAE;

extern const Loss BCE;
extern const Loss CCE;
extern const Loss HINGE;

extern const Loss KL_DIVERGENCE;



#endif