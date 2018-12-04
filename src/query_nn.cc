#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <memory>

static std::vector<std::string> Split(const std::string& s, char sep = ' ') {
  std::vector<std::string> result;

  size_t pos1 = 0;
  size_t pos2 = s.find(sep);
  while (std::string::npos != pos2) {
    result.push_back(s.substr(pos1, pos2 - pos1));

    pos1 = pos2 + 1;
    pos2 = s.find(sep, pos1);
  }
  if (pos1 != s.length()) {
    result.push_back(s.substr(pos1));
  }
  return result;
}

static void CheckStream(const std::ifstream& ifs) {
  if (!ifs.good() && !ifs.eof()) {
    std::cout << "ifstream read error, exit." << std::endl;
    exit(-1);
  }
}

typedef float real;
class Matrix {
 public:
  real* data_;
  int64_t m_;
  int64_t n_;

  Matrix() {
    m_ = 0;
    n_ = 0;
    data_ = nullptr;
  }
  Matrix(int64_t m, int64_t n) {
    m_ = m;
    n_ = n;
    data_ = new real[m * n];
  }
  Matrix(const Matrix& other) {
    m_ = other.m_;
    n_ = other.n_;
    data_ = new real[m_ * n_];
    for (int64_t i = 0; i < (m_ * n_); i++) {
      data_[i] = other.data_[i];
    }
  }
  Matrix& operator=(const Matrix& other) {
    Matrix temp(other);
    m_ = temp.m_;
    n_ = temp.n_;
    std::swap(data_, temp.data_);
    return *this;
  }
  ~Matrix() { delete[] data_; }

  inline const real& at(int64_t i, int64_t j) const {
    return data_[i * n_ + j];
  };
  inline real& at(int64_t i, int64_t j) { return data_[i * n_ + j]; };

  void zero() {
    for (int64_t i = 0; i < (m_ * n_); i++) {
      data_[i] = 0.0;
    }
  }
  void uniform(real a) {
    std::minstd_rand rng(1);
    std::uniform_real_distribution<float> uniform(-a, a);
    for (int64_t i = 0; i < (m_ * n_); i++) {
      data_[i] = uniform(rng);
    }
  }
  real l2NormRow(int64_t i) const {
    auto norm = 0.0;
    for (auto j = 0; j < n_; j++) {
      const real v = at(i, j);
      norm += v * v;
    }
    return static_cast<real>(std::sqrt(norm));
  }
  void convertColMajor() {
    real* new_data = new real[m_ * n_];
    // 遍历列
    int i = 0;
    for (int col = 0; col < n_; ++col) {
      for (int row = 0; row < m_; ++row) {
        new_data[i++] = data_[row * n_ + col];
      }
    }

    delete[] data_;
    data_ = new_data;
  }

  // l2 normlize all rows
  void normlize() {
    for (auto i = 0; i < m_; ++i) {
      const real n = l2NormRow(i);
      for (auto j = 0; j < n_; ++j) {
        at(i, j) /= n;
      }
    }
  }
};

std::vector<std::string> dict;
std::shared_ptr<Matrix> matrix;

void LoadVectors(std::string filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cout << "Open vectors_file '" << filename << "'"
              << "failed." << std::endl;
    exit(-1);
  }
  int vocab_size, dim;
  std::string line;
  std::getline(ifs, line);
  auto tokens = Split(line, ' ');
  assert(tokens.size() == 2);
  vocab_size = std::stoi(tokens[0]);
  dim = std::stoi(tokens[1]);
  assert(dim > 0);
  assert(vocab_size > 0);
  matrix = std::make_shared<Matrix>(vocab_size, dim);

  for (int i = 0; i < vocab_size; ++i) {
    std::getline(ifs, line);
    CheckStream(ifs);
    auto tokens = Split(line, ' ');
    assert((int)tokens.size() == dim + 1);
    dict.push_back(tokens[0]);
    for (int j = 0; j < dim; ++j) {
      matrix->at(i, j) = std::stof(tokens[j + 1]);
    }
  }
  matrix->normlize();
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: <vectors_file>" << std::endl;
    exit(-1);
  }
  std::cout << "Loading vectors ..." << std::endl;
  LoadVectors(argv[1]);
  std::cout << "Load vectors done." << std::endl;

  return 0;
}
