#include <assert.h>
#include <math.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

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

class Vector {
 public:
  int64_t m_;
  real* data_;

  explicit Vector(int64_t m) {
    m_ = m;
    data_ = new real[m];
  }
  Vector& operator=(const Vector& source) {
    delete[] data_;
    m_ = source.m_;

    data_ = new real[m_];

    for (int i = 0; i < m_; ++i) {
      data_[i] = source.data_[i];
    }

    return *this;
  }

  Vector(const Vector&) = delete;

  ~Vector() { delete[] data_; }

  real& operator[](int64_t i) { return data_[i]; }
  const real& operator[](int64_t i) const { return data_[i]; }

  int64_t size() const { return m_; }
  void zero() { memset(data_, 0, sizeof(real) * m_); }
  void addRow(const Matrix& A, int64_t i) {
    assert(i >= 0);
    assert(i < A.m_);
    assert(m_ == A.n_);
    for (int64_t j = 0; j < A.n_; j++) {
      data_[j] += A.at(i, j);
    }
  }
  void mul(const Matrix& A, const Vector& vec) {
    assert(A.m_ == m_);
    assert(A.n_ == vec.m_);
    for (int64_t i = 0; i < m_; i++) {
      real d = 0.0;
      for (int64_t j = 0; j < A.n_; j++) {
        d += A.at(i, j) * vec.data_[j];
      }
      data_[i] = d;
    }
  }
};

std::vector<std::string> dict;
std::map<std::string, int> word2idx;
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
    word2idx[tokens[0]] = dict.size();
    dict.push_back(tokens[0]);
    for (int j = 0; j < dim; ++j) {
      matrix->at(i, j) = std::stof(tokens[j + 1]);
    }
  }
  matrix->normlize();
}

void findNN(const std::string& queryWord, int k,
            const std::unordered_set<std::string>& banSet) {
  int vocab_size = matrix->n_;
  Vector queryVec(vocab_size);
  auto it = word2idx.find(queryWord);
  int idx = 0;
  if (it != word2idx.end()) {
    idx = it->second;
  } else {
    std::cout << "not in dict" << std::endl;
    return;
  }
  queryVec.zero();
  queryVec.addRow(*matrix, idx);

  Vector output(matrix->m_);
  output.mul(*matrix, queryVec);

  std::vector<std::pair<real, int>> heap(matrix->n_);
  for (int32_t i = 0; i < vocab_size; i++) {
    heap[i].first = output[i];
    heap[i].second = i;
  }
  std::make_heap(heap.begin(), heap.end());

  int32_t i = 0;
  size_t poped = 0;
  std::vector<std::pair<real, std::string>> nn;
  while (i < k && heap.size() > 0) {
    auto& top = heap.front();
    std::string word;
    word = dict[top.second];
    auto it = banSet.find(word);
    if (it == banSet.end()) {
      nn.push_back({top.first, word});
      i++;
    }
    pop_heap(heap.begin(), heap.end() - poped);
    ++poped;
  }
  for (auto p : nn) {
    std::cout << p.second << ": " << p.first << std::endl;
  }
}

static bool run_cmd(char* cmd) {
  pid_t pid;
  char sh[4] = "sh";
  char arg[4] = "-c";
  char* argv[] = {sh, arg, cmd, NULL};
  std::cerr << "Run command: " << cmd << std::endl;
  int status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argv, environ);
  if (status == 0) {
    std::cerr << "Child pid: " << pid << std::endl;
    if (waitpid(pid, &status, 0) != -1) {
      std::cerr << "Child exited with status " << status << std::endl;
    } else {
      std::cerr << "Child exited with status " << status
                << ", errmsg = " << strerror(errno) << std::endl;
      return false;
    }
  } else {
    std::cerr << "posix_spawn failed, errmsg = " << strerror(status)
              << std::endl;
    return false;
  }
  return true;
}

void query_nn(const std::string& query, int k,
              std::vector<std::pair<real, std::string>>& nn) {
  std::unordered_set<std::string> banSet;
  banSet.insert(query);
  int vocab_size = matrix->n_;
  Vector queryVec(vocab_size);
  auto it = word2idx.find(query);
  int idx = 0;
  if (it != word2idx.end()) {
    idx = it->second;
  } else {
    std::cout << "not in dict" << std::endl;
    return;
  }
  queryVec.zero();
  queryVec.addRow(*matrix, idx);

  Vector output(matrix->m_);
  output.mul(*matrix, queryVec);

  std::vector<std::pair<real, int>> heap(matrix->n_);
  for (int32_t i = 0; i < vocab_size; i++) {
    heap[i].first = output[i];
    heap[i].second = i;
  }
  std::make_heap(heap.begin(), heap.end());

  int32_t i = 0;
  size_t poped = 0;
  while (i < k && heap.size() > 0) {
    auto& top = heap.front();
    std::string word;
    word = dict[top.second];
    auto it = banSet.find(word);
    if (it == banSet.end()) {
      nn.push_back({top.first, word});
      i++;
    }
    std::pop_heap(heap.begin(), heap.end() - poped);
    ++poped;
  }
}

void nn_body(const std::string& filename, int k) {
  std::string resultfile = filename + ".result";
  std::ifstream ifs(filename);
  std::ofstream ofs(resultfile);
  assert(ifs.is_open());
  assert(ofs.is_open());

  std::string query;
  std::vector<std::pair<real, std::string>> nn;

  while (!ifs.eof()) {
    std::getline(ifs, query);
    if (query.empty()) {
      continue;
    }

    nn.clear();

    query_nn(query, k, nn);

    for (auto& p : nn) {
      ofs.write(p.second.data(), p.second.size());
      ofs.write(" ", 1);
      auto s = std::to_string(p.first);
      ofs.write(s.data(), s.size());
      ofs.write("\n", 1);
    }
  }
  ifs.close();
  ofs.close();
}

void dump_nn(const std::string& queryfile, int nthread, int k) {
  static char cmd[65536];
  std::string command = "split -a 3 -d -n l/" + std::to_string(nthread) + " " +
                        queryfile + " " + queryfile + ".";
  memcpy(cmd, command.data(), command.size());
  cmd[command.size()] = '\0';
  if (!run_cmd(cmd)) {
    exit(-1);
  }

  std::vector<std::thread> threads;
  char suffix[4];
  std::string catfile;
  std::string rmfile;
  for (int i = 0; i < nthread; ++i) {
    snprintf(suffix, 4, "%03d", i);
    auto filename = queryfile + "." + suffix;
    catfile += filename + ".result";
    catfile += " ";
    rmfile += filename;
    rmfile += " ";
    std::cerr << "split query file name: " << filename << std::endl;
    threads.emplace_back(&nn_body, filename, k);
  }

  for (int i = 0; i < nthread; ++i) {
    threads[i].join();
  }

  command = "cat " + catfile + " " + " > " + queryfile + ".result";
  memcpy(cmd, command.data(), command.size());
  cmd[command.size()] = '\0';
  if (!run_cmd(cmd)) {
    exit(-1);
  }

  command = "rm " + catfile + " " + rmfile;
  memcpy(cmd, command.data(), command.size());
  cmd[command.size()] = '\0';
  if (!run_cmd(cmd)) {
    exit(-1);
  }
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "Usage: <vectors_file> <k> <query_file> <nthread>"
              << std::endl;
    exit(-1);
  }
  std::cout << "Loading vectors ..." << std::endl;
  LoadVectors(argv[1]);
  std::cout << "Load vectors done." << std::endl;
  int k = std::stoi(argv[2]);
  std::string query_file = argv[3];
  int nthread = std::stoi(argv[4]);

  dump_nn(query_file, nthread, k);

  return 0;
}
