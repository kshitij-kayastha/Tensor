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
git clone https://github.com/kshitij-kayastha/tensor.git
cd tensor
```

## Usage

Example usage of the Tensor library:

```cpp
#include "tensor.hpp"

int main() {
    std::vector<size_t> shape = {2, 3};
    float elements[] = {1, 2, 3, 4, 5, 6};
    Tensor<float> tensor(elements, shape);
    tensor.display();


    Tensor<float> random_tensor = tensor_rand(shape);
    random_tensor.display();
    return 0;
}
```

### Compile and Run
```sh
g++ -o tensor_example main.cpp Tensor.cpp -std=c++17
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
For questions and support, open an issue or reach out at kshitijkayastha@gmail.com.

