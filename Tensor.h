// Author: Kshitij Kayastha
// Date: 03/13/2025


#ifndef __TENSOR_BASE_H__
#define __TENSOR_BASE_H__


#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

template <typename T>
class Tensor {
private:
    T* m_elements;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_stride;
    size_t m_size = 1;

public:
    Tensor(T* elements, std::vector<size_t> shape) : m_shape(shape) {
        for (size_t i = 0; i < m_shape.size(); ++i) {
            m_size *= shape[i];
        }
        m_elements = new T[m_size];
        for (size_t i = 0; i < m_size; ++i) {
            m_elements[i] = elements[i];
        }
        m_stride = compute_strides(m_shape);
    }
    ~Tensor() {
        delete[] m_elements;
    }

    std::vector<size_t> shape(void) const {
        return m_shape;
    }

    std::vector<size_t> stride(void) const {
        return m_stride;
    }

    size_t size(void) const {
        return m_size;
    }

    /*
    =========================
    Getter and Setter Methods
    =========================
    */


    const T at(size_t i) const {
        return m_elements[i];
    }

    const T at(const std::vector<size_t> loc) const {
        for (size_t i = 0; i < loc.size(); ++i) {
        assert(loc[i] <= m_shape[i]);
    }
    size_t idx = loc2idx(loc);
    return m_elements[idx];
    }
    
    void set(size_t i, T value) {
        if (i > m_size) {
            throw std::out_of_range("Index out of range\n");
        }
        m_elements[i] = value;
    }
    void set(const std::vector<size_t> loc, T value) {
        size_t idx = loc2idx(loc);
        m_elements[idx] = value;
    }
    
    void fill(T x) {
        for (size_t i = 0; i < m_size; ++i) {
            m_elements[i] = x;
        }
    }

    /*
    ===================
    Display Methods
    ===================
    */

    void display(void) const {
        std::vector<size_t> loc(m_shape.size(), 0);
        for (size_t c = 0; c < m_size; ++c) {
            size_t curr_sum = 0;
            for (size_t i = 0; i < m_shape.size(); ++i) {
                curr_sum += loc[i];
            }
            for (size_t i = 0; i < m_shape.size(); ++i) {
                printf((curr_sum == 0) ? "[" : " ");
                curr_sum = curr_sum - loc[i];
            }

            if constexpr (std::is_integral<T>::value) {
                printf("%4d", static_cast<int>(at(loc)));
            } else if constexpr (std::is_floating_point<T>::value) {
                printf("%8.4f", static_cast<double>(at(loc)));
            } else {
                std::cout << at(loc) << " ";
            }
            
            bool increment = true;
            size_t curr_size = 1;
            size_t total_size = 1;
            for (int i = m_shape.size()-1; i >= 0; --i) {
                if (increment) {
                    loc[i] += 1;
                    increment = false;
                }
                curr_size = curr_size * loc[i];
                total_size = total_size * m_shape[i];
                if (curr_size == total_size){
                    printf(i == (int) m_shape.size()-1 ? "  ]": "]");
                }
                if (loc[i] >= m_shape[i]) {
                    loc[i] = 0;
                    increment = true;
                    for (size_t j = 0; j < (m_shape.size()-i); ++j){
                        if (loc[i-1] < m_shape[i-1]-1) {
                            printf((c < m_size-1) ? "\n": "");
                        }
                    }
                }
            }
        }
        printf("\n");
    }

    /*
    ===================
    Scalar Math Methods
    ===================
    */
    
    void add(float x) {
        for (size_t i = 0; i < m_size; ++i) {
            set(i, at(i) + x);
        }
    }
    
    void multiply(float x) {
        for (size_t i = 0; i < m_size; ++i) {
            set(i, at(i) * x);
        }
    }
    void pow(float x) {
        for (size_t i = 0; i < m_size; ++i) {
            set(i, std::pow(at(i), x));
        }
    }

    void sqrt(void) {
        for (size_t i = 0; i < m_size; ++i) {
            set(i, std::sqrt(at(i)));
        }
    }

    void exp(float x = std::exp(1.)) {
        for (size_t i = 0; i < m_size; ++i) {
            set(i, std::pow(x, at(i)));
        }
    }

    void apply(std::function<T(T)> fn) {
        for (size_t i = 0; i < m_size; ++i) {
            set(i, fn(at(i)));
        }
    }
    
    /*
    ===================
    Tensor Math Methods
    ===================
    */

    Tensor<T> add(const Tensor<T>& tensor) const {
        std::vector<size_t> tensor_shape = tensor.shape();
        if (m_shape.size() != tensor_shape.size()) {
            throw std::out_of_range("Shape of tensor a does not match the shape of tensor b\n");
        }

        for (size_t i = 0; i < m_shape.size(); ++i) {
            if (m_shape[i] != tensor_shape[i]) {
                throw std::out_of_range("Shape of tensor a does not match the shape of tensor b\n");
            }
        }
        
        T* elements = new T[m_size];
        assert(elements != nullptr);
        for (size_t i = 0; i < m_size; ++i) {
            elements[i] = m_elements[i] + tensor.at(i);
        }

        Tensor<T> result(elements, m_shape);
        delete[] elements;
        return result;
    }

    Tensor<T> subtract(const Tensor<T>& tensor) const {
        std::vector<size_t> tensor_shape = tensor.shape();
        if (m_shape.size() != tensor_shape.size()) {
            throw std::out_of_range("Shape of tensor a does not match the shape of tensor b\n");
        }

        for (size_t i = 0; i < m_shape.size(); ++i) {
            if (m_shape[i] != tensor_shape[i]) {
                throw std::out_of_range("Shape of tensor a does not match the shape of tensor b\n");
            }
        }
        
        T* elements = new T[m_size];
        assert(elements != nullptr);
        for (size_t i = 0; i < m_size; ++i) {
            elements[i] = m_elements[i] - tensor.at(i);
        }

        Tensor<T> result(elements, m_shape);
        delete[] elements;
        return result;
    }

    // Tensor<T> dot(const Tensor<T>& tensor) const;

    /*
    =================
    Reduction Methods
    =================
    */

    Tensor<T> sum(int axis = -1) {
        // TODO
        assert(axis < 10);
        T total = 0;
        for (size_t i = 0; i < m_size; ++i) {
            total += m_elements[i];    
        }
        T elements[1] = {total};
        return Tensor<T>(elements, {1, 1});
    }

    // Tensor max(int axis = -1)

    // Tensor min(int axis = -1) 

    // Tensor mean(int axis = -1)

    /*
    ======================
    Transformation Methods
    ======================
    */

    void reshape(const std::vector<size_t> shape) {
        std::vector<size_t> strides = compute_strides(shape);
        size_t size = strides[0] * shape[0];
        if (m_size != size) {
            throw std::out_of_range("Invalid shape provided\n");
        }
        m_shape = shape;
        m_stride = compute_strides(m_shape);
    }

    void flatten(void) {
        std:: vector<size_t> shape = {1, m_size};
        m_shape = shape;
    }
    
    void transpose(void) {
        // Slow algorithm for transposing large tensors
    
        T* elements = new T[m_size];
        assert(elements != nullptr);

        std::vector<size_t> shape = m_shape;
        std::reverse(shape.begin(), shape.end());

        for (size_t i = 0; i < m_size; ++i) {
            std::vector<size_t> loc = idx2loc(i);
            std::reverse(loc.begin(), loc.end());

            size_t idx = loc2idx(loc, shape);
            elements[idx] = m_elements[i];
        }

        delete[] m_elements;
        m_elements = elements;
        m_shape = shape;
        m_stride = compute_strides(m_shape);
    }

    /*
    =============
    Other Methods
    =============
    */

    Tensor<T> clone(void) const {
        return Tensor<T>(m_elements, m_shape);
    }

private:

    /*
    =========================
    Loc to Idx to Loc Methods
    =========================
    */

    std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const {
        std::vector<size_t> strides(shape.size());
        strides.back() = 1;
        for (int i = shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    size_t loc2idx(const std::vector<size_t>& loc) const {
        if (m_shape.size() != loc.size()) {
            throw std::out_of_range("Input shape is invalid\n");
        }
        size_t idx = 0;
        for (size_t i = 0; i < m_shape.size(); ++i) {
            if (loc[i] >= m_shape[i]) {
                throw std::out_of_range("Index out of bounds\n");
            }
            idx += loc[i] * m_stride[i];
        }
        return idx;
    }

    size_t loc2idx(const std::vector<size_t>& loc, const std::vector<size_t>& shape) const {
        if (m_shape.size() != loc.size()) {
            throw std::out_of_range("Input shape is invalid\n");
        }
        std::vector<size_t> strides = compute_strides(shape);
        size_t idx = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (loc[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds\n");
            }
            idx += loc[i] * strides[i];
        }
        return idx;
    }

    std::vector<size_t> idx2loc(size_t idx) const {
        if (idx >= m_size) {
            throw std::out_of_range("Index out of bounds\n");
        }
        std::vector<size_t> loc(m_shape.size(), 0);
        for (size_t i = 0; i < m_shape.size(); ++i) {
            loc[i] = idx / m_stride[i];
            idx %= m_stride[i];
        }
        return loc;
    }

    std::vector<size_t> idx2loc(size_t idx, const std::vector<size_t>& shape) const {
        std::vector<size_t> strides = compute_strides(shape);
        size_t size = strides[0] * shape[0];
        if (idx >= size) {
            throw std::out_of_range("Index out of bounds\n");
        }

        std::vector<size_t> loc(shape.size(), 0);
        for (size_t i = 0; i < shape.size(); ++i) {
            loc[i] = idx / strides[i];
            idx %= strides[i];
        }
        return loc;
    }
};


/*
===========
Tensor APIs
===========
*/

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

Tensor<float> tensor_rand(size_t size, float low, float high) {
    if (size < 0) {
        throw std::invalid_argument("Trying to create tensor with negative dimension\n");
    }

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


#endif // TENSOR_H
