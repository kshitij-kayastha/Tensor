// Author: Kshitij Kayastha
// Date: 03/13/2025


#include "TensorBase.h"


template<typename T>
Tensor<T> tensor(T* data, size_t size) {
    return Tensor<T>(data, {1, size});
}

template<typename T>
Tensor<T> tensor(std::vector<T> data) {
    T elements[data.size()];
    for (size_t i = 0; i < data.size(); ++i) {
        elements[i] = data[i];
    }
    return Tensor<T>(elements, {1, data.size()});
}

template<typename T>
Tensor<T> tensor_arange(size_t stop) {
    assert (stop > 0);
    std::vector<size_t> shape = {1, stop};
    T elements[stop];
    for (size_t i = 0; i < stop; ++i) {
        elements[i] = i;
    }
    return Tensor<T>(elements, shape);
}

template <typename T>
Tensor<T> tensor_arange(int start, int stop, int stride) {
    if (stride == 0) {
        throw std::invalid_argument("Stride must be nonzero\n");
    }
    if (stop < start){
        if (stride >= 0) {
            throw std::invalid_argument("Stride must be negative\n");
        }
    } else if (start < stop) {
        if (stride < 0) {
            throw std::invalid_argument("Stride must be positive\n");
        }
    }

    size_t size = std::ceil(((float) stop - start) / stride);
    std::vector<size_t> shape = {1, size};
    T elements[size];
    for (size_t i = 0; i < size; ++i) {
        elements[i] = (T) i * stride + start;
    }
    return Tensor<T>(elements, shape);
}

// TODO: overload for int and other types
Tensor<float> tensor_rand(size_t size, float low, float high) {
    assert(size >= 0 && "Trying to create tensor with negative dimension\n");

    float elements[size];
    for (size_t i = 0; i < size; ++i) {
        elements[i] = (((float) rand() * (high - low)) / (float) RAND_MAX) + low;
    }
    return Tensor<float>(elements, {1,size});
}

Tensor<float> tensor_rand(std::vector<size_t> shape, float low, float high) {
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            throw std::invalid_argument("Trying to create tensor with negative dimension\n");
        }
        size *= shape[i];
    }
    constexpr size_t SIZE_THRESHOLD = 1024;

    if (size <= SIZE_THRESHOLD) {
        // Stack allocation for small tensors
        float elements[size];
        for (size_t i = 0; i < size; ++i) {
            elements[i] = (((float) rand() * (high - low)) / (float) RAND_MAX) + low;
        }
        return Tensor<float>(elements, shape);
    } else{
        // Heap allocation for large tensors
        float* elements = new float[size];
        for (size_t i = 0; i < size; ++i) {
            elements[i] = (((float) rand() * (high - low)) / (float) RAND_MAX) + low;
        }
        Tensor<float> tensor(elements, shape);
        delete[] elements;
        return tensor;
    }
}

template <typename T>
Tensor<T> tensor_identity(size_t size) {
    assert(size > 0);

    T elements[size*size];
    size_t c = size;
    for (size_t i = 0; i < size*size; ++i) {
        elements[i] = (c == size) ? 1 : 0;
        c = (c == size) ? 0 : ++c;
    }
    return Tensor<T>(elements, {size, size});
}

template <typename T>
Tensor<T> tensor_zeroes(size_t size) {
    if (size < 0){
        throw std::invalid_argument("Trying to create tensor with negative size\n");
    }

    T elements[size];
    for (size_t i = 0; i < size; ++i) {
        elements[i] = 0.f;
    }
    return Tensor<T>(elements, {1,size});
}

template <typename T>
Tensor<T> tensor_zeroes(std::vector<size_t> shape) {
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            throw std::invalid_argument("Trying to create tensor with negative dimension\n");
        }
        size *= shape[i];
    }

    T elements[size];
    for (size_t i = 0; i < size; ++i) {
        elements[i] = 0.f;
    }
    return Tensor<T>(elements, shape);
}

template <typename T>
Tensor<float> tensor_ones(size_t size) {
    if (size < 0){
        throw std::invalid_argument("Trying to create tensor with negative size\n");
    }

    float elements[size];
    for (size_t i = 0; i < size; ++i) {
        elements[i] = 1.f;
    }
    return Tensor<float>(elements, {1,size});
}

template <typename T>
Tensor<T> tensor_ones(std::vector<size_t> shape) {
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            throw std::invalid_argument("Trying to create tensor with negative dimension\n");
        }
        size *= shape[i];
    }

    T elements[size];
    for (size_t i = 0; i < size; ++i) {
        elements[i] = 1.f;
    }
    return Tensor<T>(elements, shape);
}

template <typename T>
Tensor<T> tensor_eye(size_t rows, size_t cols, int offset) {
    if (cols == 0) {
        cols = rows;
    }
    if (rows <= 0) {
        throw std::invalid_argument("Trying to create tensor with negative rows\n");
    }
    if (cols <= 0) {
        throw std::invalid_argument("Trying to create tensor with negative columns\n");
    }
    // TODO: check if negative offset shift diagonal to the left

    T elements[rows*cols];
    size_t r;
    size_t c;
    for (size_t i = 0; i < rows*cols; ++i) {
        r = i / cols;
        c = i % cols;
        elements[i] = (c - r == offset) ? 1.f : 0.f; 
    }
    return Tensor<T>(elements, {rows, cols});
}