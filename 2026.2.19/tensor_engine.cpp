#include <iostream>
#include <print>
#include <stdexcept>
#include <vector>

class Tensor {
private:
  std::vector<float> data;
  int rows;
  int cols;

public:
  Tensor(int r, int c) : rows(r), cols(c) {
    data.resize(rows * cols);
    data.assign(rows * cols, 0.0f);
  }

  float &operator()(int r, int c) { return data[r * cols + c]; }

  float operator()(int r, int c) const { return data[r * cols + c]; }

  Tensor transpose() const {
    Tensor result(cols, rows);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        result(j, i) = (*this)(i, j);
      }
    }
    return result;
  }

  Tensor matmul(Tensor &other) {
    if (this->cols != other.rows) {
      throw std::invalid_argument("两矩阵维度不匹配，无法相乘");
    }
    Tensor BT = other.transpose();
    Tensor result(this->rows, other.cols);
    for (int i = 0; i < this->rows; ++i) {
      for (int j = 0; j < other.cols; ++j) {
        for (int k = 0; k < this->cols; ++k) {
          result(i, j) += (*this)(i, k) * BT(j, k);
        }
      }
    }
    return result;
  }

  Tensor operator+(const Tensor &other) const {
    if (!(this->cols == other.cols &&
          (this->rows == other.rows || other.rows == 1))) {
      throw std::invalid_argument("两矩阵维度不匹配，无法相加");
    }
    Tensor result(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
      for (int j = 0; j < this->cols; ++j) {
        result(i, j) = (*this)(i, j) + other(other.rows == 1 ? 0 : i, j);
      }
    }
    return result;
  }

  Tensor operator-(const Tensor &other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
      throw std::invalid_argument("两矩阵维度不匹配，无法相减");
    }
    Tensor result(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
      for (int j = 0; j < this->cols; ++j) {
        result(i, j) = (*this)(i, j) - other(i, j);
      }
    }
    return result;
  }

  Tensor operator*(float scalar) const {
    Tensor result(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
      for (int j = 0; j < this->cols; ++j) {
        result(i, j) = (*this)(i, j) * scalar;
      }
    }
    return result;
  }

  void print() const {
    for (int i = 0; i < this->rows; ++i) {
      for (int j = 0; j < this->cols; ++j) {
        std::cout << (*this)(i, j) << "\t";
      }
      std::cout << std::endl;
    }
  }
};

int main() {
  Tensor A(2, 2);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;

  Tensor Bias(1, 2);
  Bias(0, 0) = 10;
  Bias(0, 1) = 20;

  std::cout << "Testing Broadcasting: A + Bias\n";
  try {
    Tensor result = A + Bias;
    result.print();
  } catch (const std::exception &e) {
    std::cout << "Exception caught: " << e.what() << "\\n";
  }

  return 0;
}