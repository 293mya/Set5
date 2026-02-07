

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

class RandomStreamGen {
public:
  RandomStreamGen(std::uint64_t seed = std::random_device{}())
      : rng_(seed), len_dist_(1, 30),
        char_dist_(0, static_cast<int>(alphabet().size()) - 1) {}

  std::vector<std::string> generate(std::size_t n) {
    std::vector<std::string> res;
    res.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      res.push_back(random_string());
    }
    return res;
  }

private:
  std::string random_string() {
    std::size_t len = static_cast<std::size_t>(len_dist_(rng_));
    std::string s;
    s.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
      s.push_back(alphabet()[static_cast<std::size_t>(char_dist_(rng_))]);
    }
    return s;
  }

  static const std::string &alphabet() {
    static const std::string chars = "abcdefghijklmnopqrstuvwxyz"
                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                     "0123456789-";
    return chars;
  }

  std::mt19937_64 rng_;
  std::uniform_int_distribution<int> len_dist_;
  std::uniform_int_distribution<int> char_dist_;
};

class HashFuncGen {
public:
  explicit HashFuncGen(const std::string &salt = "default_salt")
      : salt_(salt) {}

  std::uint32_t operator()(const std::string &s) const {
    static std::hash<std::string> hasher;
    std::size_t h = hasher(s + salt_);

    std::uint32_t low = static_cast<std::uint32_t>(h & 0xffffffffu);
    std::uint32_t high = static_cast<std::uint32_t>((h >> 32) & 0xffffffffu);
    return low ^ high;
  }

private:
  std::string salt_;
};

class HyperLogLog {
public:
  explicit HyperLogLog(unsigned B_bits, const HashFuncGen &h)
      : B_(B_bits), m_(1u << B_bits), registers_(m_, 0), hash_(h) {}

  void add(const std::string &value) {
    std::uint32_t x = hash_(value);

    std::uint32_t idx = x >> (32u - B_);

    std::uint32_t w = x << B_;

    std::uint8_t rho;
    if (w != 0) {

      rho = static_cast<std::uint8_t>(__builtin_clz(w) + 1);
    } else {

      rho = static_cast<std::uint8_t>((32u - B_) + 1u);
    }

    if (rho > registers_[idx]) {
      registers_[idx] = rho;
    }
  }

  double estimate() const {
    double alpha_m = alpha_for_m(m_);

    double sum = 0.0;
    for (std::uint8_t reg : registers_) {
      sum += std::ldexp(1.0, -static_cast<int>(reg));
    }

    double Z = 1.0 / sum;
    double raw_estimate = alpha_m * m_ * m_ * Z;

    std::size_t V = std::count(registers_.begin(), registers_.end(),
                               static_cast<std::uint8_t>(0));
    if (V != 0) {
      double linear_counting =
          m_ * std::log(static_cast<double>(m_) / static_cast<double>(V));
      if (linear_counting <= 5.0 * m_) {
        return linear_counting;
      }
    }

    return raw_estimate;
  }

  unsigned B_bits() const { return B_; }
  std::size_t m() const { return m_; }

private:
  static double alpha_for_m(std::size_t m) {

    if (m == 16)
      return 0.673;
    if (m == 32)
      return 0.697;
    if (m == 64)
      return 0.709;
    return 0.7213 / (1.0 + 1.079 / static_cast<double>(m));
  }

  unsigned B_;
  std::size_t m_;
  std::vector<std::uint8_t> registers_;
  const HashFuncGen &hash_;
};

struct StepResult {
  std::size_t stream_id;
  std::size_t step_index;
  std::size_t items_processed;
  double true_F0;
  double hll_estimate;
};

int main() {

  const unsigned B_bits = 10;
  const std::size_t m = 1u << B_bits;

  std::cout << "Эксперимент HyperLogLog\n";
  std::cout << "B = " << B_bits << ", m = " << m << "\n";
  std::cout << "Теоретическое относительное σ примерно равно 1.04/sqrt(m).\n";
  std::cout << "В задании также нужно проверить sqrt(1.04/2^B) и "
               "sqrt(1.3/2^B).\n\n";

  const std::size_t num_streams = 5;
  const std::size_t stream_length = 100000;
  const std::size_t step_percent = 5;

  if (step_percent == 0 || step_percent > 100 || (100 % step_percent) != 0) {
    std::cerr << "step_percent должен делить 100\n";
    return 1;
  }

  const std::size_t num_steps = 100 / step_percent;
  const std::size_t step_size = stream_length / num_steps;

  std::uint64_t base_seed = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());

  std::ofstream csv_all("hll_results_all_streams.csv");
  csv_all << "stream_id,step_index,items_processed,true_F0,hll_estimate\n";

  for (std::size_t s = 0; s < num_streams; ++s) {
    std::uint64_t seed = base_seed + s * 1000;
    RandomStreamGen stream_gen(seed + 1);

    HashFuncGen hash("hll_salt_" + std::to_string(seed + 2));
    HyperLogLog hll(B_bits, hash);

    std::vector<std::string> stream = stream_gen.generate(stream_length);

    std::unordered_set<std::string> distinct;
    distinct.reserve(stream_length * 2);

    std::size_t items_processed = 0;

    std::string filename = "hll_stream_" + std::to_string(s) + ".csv";
    std::ofstream csv_stream(filename);
    csv_stream << "step_index,items_processed,true_F0,hll_estimate\n";

    for (std::size_t step = 0; step < num_steps; ++step) {
      std::size_t until = (step + 1) * step_size;
      if (until > stream_length) {
        until = stream_length;
      }
      for (; items_processed < until; ++items_processed) {
        const std::string &val = stream[items_processed];
        distinct.insert(val);
        hll.add(val);
      }

      double true_f0 = static_cast<double>(distinct.size());
      double est = hll.estimate();

      csv_stream << step << ',' << items_processed << ',' << true_f0 << ','
                 << est << '\n';

      csv_all << s << ',' << step << ',' << items_processed << ',' << true_f0
              << ',' << est << '\n';
    }

    csv_stream.close();
  }

  csv_all.close();

  std::cout << "Готово. Сгенерированы CSV по потокам: hll_stream_*.csv\n";
  std::cout << "и агрегированный файл hll_results_all_streams.csv.\n";

  return 0;
}
