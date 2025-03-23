// Author: Kshitij Kayastha
// Date: 03/13/2025

#include <time.h>
#include <vector>
#include "../Tensor.h"

float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

int main(void) {
    srand(time(NULL));

    auto t1 = tensor_arange<float>(10);
    printf("Original\n");
    t1.display();
    printf("\n");
    
    printf("Sum\n");
    t1.sum().display();
    printf("\n");

    printf("Apply sigmoid\n");
    t1.apply(sigmoid);
    t1.display();
    
    return 0;
}