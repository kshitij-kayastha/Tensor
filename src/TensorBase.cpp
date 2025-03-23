// Author: Kshitij Kayastha
// Date: 03/13/2025

#include "TensorBase.h"

template <typename T>
Tensor<T>::Tensor(T* elements, std::vector<size_t> shape) : m_shape(shape) {
    for (size_t i = 0; i < m_shape.size(); ++i) {
        m_size *= shape[i];
    }
    m_elements = new T[m_size];
    for (size_t i = 0; i < m_size; ++i) {
        m_elements[i] = elements[i];
    }
    m_stride = compute_strides(m_shape);
}

template <typename T>
Tensor<T>::~Tensor() {
    delete[] m_elements;
}

template <typename T>
std::vector<size_t> Tensor<T>::shape(void) const {
    return m_shape;
}

template <typename T>
std::vector<size_t> Tensor<T>::stride(void) const {
    return m_stride;
}

template <typename T>
size_t Tensor<T>::size(void) const {
    return m_size;
}

template <typename T>
const T Tensor<T>::at(size_t i) const {
    return m_elements[i];
}

template <typename T>
const T Tensor<T>::at(const std::vector<size_t> loc) const {
    for (size_t i = 0; i < loc.size(); ++i) {
        assert(loc[i] <= m_shape[i]);
    }
    size_t idx = loc2idx(loc);
    return m_elements[idx];
}

template <typename T>
void Tensor<T>::set(size_t i, T value) {
    m_elements[i] = value;
}

template <typename T>
void Tensor<T>::set(const std::vector<size_t> loc, T value) {
    size_t idx = loc2idx(loc);
    m_elements[idx] = value;
}

template <typename T>
void Tensor<T>::fill(T x) {
    for (size_t i = 0; i < m_size; ++i) {
        m_elements[i] = x;
    }
}

template <typename T>
void Tensor<T>::display(void) const {
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

template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& tensor) const {
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

template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T>& tensor) const {
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

template <typename T>
void Tensor<T>::add(float x) {
    for (size_t i = 0; i < m_size; ++i) {
        set(i, at(i) + x);
    }
}

template <typename T>
void Tensor<T>::multiply(float x) {
    for (size_t i = 0; i < m_size; ++i) {
        set(i, at(i) * x);
    }
}

template <typename T>
void Tensor<T>::pow(float x) {
    for (size_t i = 0; i < m_size; ++i) {
        set(i, std::pow(at(i), x));
    }
}

template <typename T>
void Tensor<T>::sqrt(void) {
    for (size_t i = 0; i < m_size; ++i) {
        set(i, std::sqrt(at(i)));
    }
}

template <typename T>
void Tensor<T>::exp(float x) {
    for (size_t i = 0; i < m_size; ++i) {
        set(i, std::pow(x, at(i)));
    }
}

template <typename T>
Tensor<T> Tensor<T>::sum(int axis) {
    assert(axis < 10);
    T elements[1];
    for (size_t i = 0; i < m_size; ++i) {
        elements[0] += m_elements[i];    
    }
    return Tensor<T>(elements, {1, 1});
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t> shape) {
    auto strides = compute_strides(shape);
    size_t size = strides[0] * shape[0];
    if (m_size != size) {
        throw std::out_of_range("Invalid shape provided\n");
    }
    m_shape = shape;
    m_stride = compute_strides(m_shape);
}

template <typename T>
void Tensor<T>::flatten(void) {
    std:: vector<size_t> shape = {1, m_size};
    m_shape = shape;
}

template <typename T>
void Tensor<T>::transpose(void) {
        
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

template <typename T>
Tensor<T> Tensor<T>::clone(void) const {
    return Tensor<T>(m_elements, m_shape);
}

template <typename T>
void Tensor<T>::apply(std::function<T(T)> fn) {
    for (size_t i = 0; i < m_size; ++i) {
        set(i, fn(at(i)));
    }
}

/* 
===============
Private Methods
===============
*/ 

template <typename T>
std::vector<size_t> Tensor<T>::compute_strides(const std::vector<size_t>& shape) const {
    std::vector<size_t> strides(shape.size());
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template <typename T>
size_t Tensor<T>::loc2idx(const std::vector<size_t>& loc) const {
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

template <typename T>
size_t Tensor<T>::loc2idx(const std::vector<size_t>& loc, const std::vector<size_t>& shape) const {
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

template <typename T>
std::vector<size_t> Tensor<T>::idx2loc(size_t idx) const {
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

template <typename T>
std::vector<size_t> Tensor<T>::idx2loc(size_t idx, const std::vector<size_t>& shape) const {
    auto strides = compute_strides(shape);
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

template class Tensor<int>;
template class Tensor<unsigned int>;
template class Tensor<short int>;
template class Tensor<unsigned short int >;
template class Tensor<long int>;
template class Tensor<long long int>;
template class Tensor<unsigned long long int>;
template class Tensor<size_t>;
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<long double>;