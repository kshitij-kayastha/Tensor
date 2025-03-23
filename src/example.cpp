// Author: Kshitij Kayastha
// Date: 03/13/2025

#include <time.h>
#include <vector>
#include "Tensor.h"

float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

float square(float x) {
    return x*x;
}

int main(void) {
    srand(time(NULL));

    Tensor<float> t1 = tensor_rand({4, 3, 2});
    printf("Actual Data\n");
    t1.display();
    printf("\n");
    
    printf("Reshape\n");
    t1.reshape({3,4,2});
    t1.display();
    printf("Done\n");







    return 0;
}