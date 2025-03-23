// Author: Kshitij Kayastha
// Date: 03/13/2025

#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include <iostream>
#include <cassert>
#include <functional>
#include <cmath>
#include <type_traits>

template <typename T>
class Tensor {
private:
    T* m_elements;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_stride;
    size_t m_size = 1;

public:
    Tensor(T* elements, std::vector<size_t> shape);
    ~Tensor();

    std::vector<size_t> shape(void) const;
    std::vector<size_t> stride(void) const;
    size_t size(void) const;

    // Getter and Setter Methods
    const T at(size_t i) const;
    const T at(const std::vector<size_t> loc) const;
    void set(size_t i, T value);
    void set(const std::vector<size_t> loc, T value);
    void fill(T x);

    // Display Methods
    void display(void) const;

    // Tensor Math Methods
    Tensor<T> add(const Tensor<T>& tensor) const;
    Tensor<T> subtract(const Tensor<T>& tensor) const;
    // Tensor<T> dot(const Tensor<T>& tensor) const;

    // Scalar Math Methods
    void add(float x);
    void multiply(float x);
    void pow(float x);
    void sqrt(void);
    void exp(float x = std::exp(1.));

    // Reduction Methods
    // Tensor sum(int axis = -1)
    // Tensor max(int axis = -1)
    // Tensor min(int axis = -1) 
    // Tensor mean(int axis = -1)

    // Transformation Methods
    void reshape(const std::vector<size_t> shape);
    void flatten(void);
    void transpose(void);

    // Other Methods
    Tensor<T> clone(void) const;
    void apply(std::function<T(T)> fn);

private:
    std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const;
    size_t loc2idx(const std::vector<size_t>& loc) const;
    size_t loc2idx(const std::vector<size_t>& loc, const std::vector<size_t>& shape) const;
    std::vector<size_t> idx2loc(size_t idx) const;
    std::vector<size_t> idx2loc(size_t idx, const std::vector<size_t>& shape) const;
};

extern template class Tensor<int>;
extern template class Tensor<size_t>;
extern template class Tensor<float>;
extern template class Tensor<double>;

// Tensor APIs

template <typename T>
Tensor<T> tensor(T* data, size_t size);

template<typename T>
Tensor<T> tensor(std::vector<T> data);

template<typename T>
Tensor<T> tensor_arange(size_t stop);

template<typename T>
Tensor<T> tensor_arange(int start, int stop, int stride);

Tensor<float> tensor_rand(size_t size, float low = 0., float high = 1.);

Tensor<float> tensor_rand(std::vector<size_t> shape, float low = 0., float high = 1.);

template <typename T>
Tensor<T> tensor_identity(size_t size);

template <typename T>
Tensor<T> tensor_zeroes(size_t size);

template <typename T>
Tensor<T> tensor_zeroes(std::vector<size_t> shape);

template <typename T>
Tensor<float> tensor_ones(size_t size);

template <typename T>
Tensor<T> tensor_ones(std::vector<size_t> shape);

template <typename T>
Tensor<T> tensor_eye(size_t rows, size_t cols=0, int offset = 0);

#endif // TENSOR_H
