// Author: Kshitij Kayastha
// Date: 03/13/2025


#ifndef __TENSOR_H__
#define __TENSOR_H__


#include "TensorBase.h"


template <typename T>
Tensor<T> tensor(T* data, size_t size);

template<typename T>
Tensor<T> tensor(std::vector<T> data);

template<typename T>
Tensor<T> tensor_arange(size_t stop);

template <typename T>
Tensor<T> tensor_arange(int start, int stop, int stride = 1);

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

#include "Tensor.cpp"

#endif