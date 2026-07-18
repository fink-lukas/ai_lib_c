#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
// #include "layer.h"
#include "activations.h"
#include "network.h"

uint32_t flip_endian(uint32_t num)
{
    return ((num >> 24) & 0x000000FF) |
           ((num >> 8) & 0x0000FF00) |
           ((num << 8) & 0x00FF0000) |
           ((num << 24) & 0xFF000000);
}

float polynomial(float x)
{
    // f(x) = 0.002 x^(4)-0.01f x^(3)-0.06f x^(2)-2
    // return 0.002f * x * x * x * x - 0.01f * x * x * x -0.06f * x * x - 2.0f;

    // f(x)=0.02 x^(4)+0.01 x^(3)-0.6 x^(2)+0.3 x -4
    return 0.02f * x * x * x * x + 0.01f * x * x * x - 0.6 * x * x + 0.3f * x - 4.0f;
}

float *load_mnist_images(const char *filename, int *out_image_count)
{
    FILE *file = fopen(filename, "rb");

    if (!file)
    {
        fprintf(stderr, "ERROR: Could not open file %s\n", filename);
        return NULL;
    }

    uint32_t identifier = 0;
    fread(&identifier, sizeof(uint32_t), 1, file);
    identifier = flip_endian(identifier);
    if (identifier != 2051)
    {
        fprintf(stderr, "ERROR: Wrong File. Not image tensor file expected\n");
        fclose(file);
        return NULL;
    }

    uint32_t numberOfImages = 0;
    fread(&numberOfImages, sizeof(uint32_t), 1, file);
    numberOfImages = flip_endian(numberOfImages);
    *out_image_count = (int)numberOfImages;

    uint32_t rows = 0;
    fread(&rows, sizeof(uint32_t), 1, file);
    rows = flip_endian(rows);

    uint32_t cols = 0;
    fread(&cols, sizeof(uint32_t), 1, file);
    cols = flip_endian(cols);

    uint32_t data_points = numberOfImages * rows * cols;
    unsigned char *raw_pixels = malloc(data_points * sizeof(unsigned char));

    if (!raw_pixels)
    {
        fprintf(stderr, "ERROR: Memory allocation failed for raw pixels\n");
        fclose(file);
        return NULL;
    }

    fread(raw_pixels, sizeof(unsigned char), data_points, file);

    float *returnArray = malloc(data_points * sizeof(float));
    if (!returnArray)
    {
        fprintf(stderr, "ERROR: Memory allocation failed for return array\n");
        free(raw_pixels);
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < (int)data_points; i++)
    {
        returnArray[i] = (float)raw_pixels[i] / 255.0f;
    }
    free(raw_pixels);
    fclose(file);
    return returnArray;
}

float *load_mnist_labels(const char *filename, int *out_label_count)
{
    FILE *file = fopen(filename, "rb");

    if (!file)
    {
        fprintf(stderr, "ERROR: Could not open file %s\n", filename);
        return NULL;
    }

    uint32_t identifier = 0;
    fread(&identifier, sizeof(uint32_t), 1, file);
    identifier = flip_endian(identifier);
    if (identifier != 2049)
    {
        fprintf(stderr, "ERROR: Wrong File. Not image tensor file expected\n");
        fclose(file);
        return NULL;
    }

    uint32_t numberOfLabels = 0;
    fread(&numberOfLabels, sizeof(uint32_t), 1, file);
    numberOfLabels = flip_endian(numberOfLabels);
    *out_label_count = (int)numberOfLabels;

    unsigned char *raw_data = malloc(numberOfLabels * sizeof(unsigned char));
    if (raw_data == NULL)
    {
        fprintf(stderr, "ERROR: Memory allocation failed for raw data\n");
        fclose(file);
        return NULL;
    }

    fread(raw_data, sizeof(unsigned char), numberOfLabels, file);

    uint32_t total_floats = numberOfLabels * 10;
    float *returnArray = calloc(total_floats, sizeof(float));

    if (returnArray == NULL)
    {
        fprintf(stderr, "ERROR: Memory allocation failed for return array\n");
        fclose(file);
        free(raw_data);
        return NULL;
    }

    for (int i = 0; i < (int)numberOfLabels; i++)
    {
        returnArray[i * 10 + (int)raw_data[i]] = 1.0f;
    }

    free(raw_data);
    fclose(file);

    return returnArray;
}

void print_guess(float *prediction, float *targetf)
{
    int target = 0;

    for (int i = 0; i < 10; i++)
    {
        if (targetf[i] == 1.0f)
        {
            target = i;
            break;
        }
    }

    float max_prob = -1.0f;
    int guess = 0;
    for (int i = 0; i < 10; i++)
    {
        if (prediction[i] > max_prob)
        {
            max_prob = prediction[i];
            guess = i;
        }
    }

    printf("Model guess: %d        Probability: %f        Target: %d\n", guess, max_prob, target);
}

int main()
{
    int image_count = 0;
    int label_count = 0;

    float *images = load_mnist_images("data/train-images.idx3-ubyte", &image_count);

    if (!images)
    {
        printf("Failed to load images. Check your file path string and terminal location.\n");
        return 1;
    }

    float *labels = load_mnist_labels("data/train-labels.idx1-ubyte", &label_count);
    if (!labels)
    {
        printf("Failed to load labels. Check your file path string and terminal location.\n");
        return 1;
    }

    if (image_count != label_count)
    {
        printf("ERROR when reading data: images and labels not same length\n");
        return 1;
    }

    printf("READING IMAGES AND LABELS COMPLETE\n\n");

    int layout[] = {784, 128, 10};
    Activation activations[] = {RELU, SIGMOID};

    Network *net = network_create(layout, activations, 3);
    printf("NETWORK INITIALIZED\n");

    int num_epochs = 5;
    float learning_rate = 0.001f;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        printf("EPOCH %d \n", epoch);
        for (int i = 0; i < image_count; i++)
        {
            float *current_image = &images[i * 784];
            float *current_label = &labels[i * 10];

            float prediction[10];
            network_forward(net, current_image, prediction);

            float loss_gradient[10];

            for (int j = 0; j < 10; j++)
            {
                float pred = prediction[j];
                if (pred < 1e-7f)
                    pred = 1e-7f;
                if (pred > 1.0f - 1e-7f)
                    pred = 1.0f - 1e-7f;

                loss_gradient[j] = (pred - current_label[j]) / (pred * (1.0f - pred));
            }

            network_backward(net, loss_gradient, learning_rate);

            if (i % 5000 == 0)
            {
                printf("%d -> ", i);
                print_guess(prediction, current_label);
            }
        }
    }

    printf("TRAINING COMPLETE");
    network_destroy(net);
    free(images);
    free(labels);
    return 0;
}