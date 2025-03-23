# Tensor

A simple, header-only tensor library in C++ for multi-dimensional arrays, supporting basic operations such as element-wise addition, scalar multiplication, and tensor reshaping.

## Features

- Multi-dimensional tensor support
- Element-wise operations
- Broadcasting support (WIP)
- Reduction methods (WIP)
- Slice manipulation (WIP)

## Installation

Clone the repository and include `tensor.hpp` and `tensor.cpp` in your project.

```sh
git clone https://github.com/kshitij-kayastha/Tensor.git
cd Tensor
```

## Usage

Example usage of the Tensor library:

```cpp
#include <time.h>
#include <vector>
#include "Tensor.h"

float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

int main() {
    srand(time(NULL));

    Tensor<float> t1 = tensor_rand({4, 3, 2}, -10, 10);
    printf("Original\n");
    t1.display();
    printf("\n");
    
    printf("Reshape\n");
    t1.reshape({3,2,4});
    t1.display();
    printf("\n");

    printf("Apply sigmoid\n");
    t1.apply(sigmoid);
    t1.display();
}
```

### Compile and Run

```sh
g++ -o tensor_example main.cpp TensorBase.cpp -std=c++17
./tensor_example
```

## Contributing

Contributions are welcome! If you'd like to add features or fix bugs:

1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions and support, open an issue or reach out at <kshitijkayastha@gmail.com>.
