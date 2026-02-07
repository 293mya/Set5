// Stage 4: HyperLogLog improved (memory-optimized registers packed into 6 bits)
// Baseline (uint8_t registers) is kept for a fair side-by-side comparison.
// Compile: g++ -std=c++17 -O2 hyperloglog_stage4.cpp -o hll4

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

// ---------------- RandomStreamGen ----------------

class RandomStreamGen {
public:
  RandomStreamGen(std::uint64_t seed = std::random_device{}())
      : rng_(seed), len_dist_(1, 30),
        char_dist_(0, static_cast<int>(alphabet().size()) - 1) {}

  std::vector<std::string> generate(std::size_t n) {
    std::vector<std::string> res;
    res.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
      res.push_back(random_string());
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

// ---------------- HashFuncGen (std::hash + salt) ----------------

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

// ---------------- HyperLogLog baseline (uint8_t registers) ----------------

class HyperLogLogBase {
public:
  explicit HyperLogLogBase(unsigned B_bits, const HashFuncGen &h)
      : B_(B_bits), m_(1u << B_bits), registers_(m_, 0), hash_(h) {}

  void add(const std::string &value) {
    std::uint32_t x = hash_(value);

    std::uint32_t idx = x >> (32u - B_);
    std::uint32_t w = x << B_;

    std::uint8_t rho;
    if (w != 0)
      rho = static_cast<std::uint8_t>(__builtin_clz(w) + 1);
    else
      rho = static_cast<std::uint8_t>((32u - B_) + 1u);

    if (rho > registers_[idx])
      registers_[idx] = rho;
  }

  double estimate() const {
    const double alpha_m = alpha_for_m(m_);

    double sum = 0.0;
    for (std::uint8_t reg : registers_) {
      sum += std::ldexp(1.0, -static_cast<int>(reg));
    }
    double Z = 1.0 / sum;
    double raw = alpha_m * m_ * m_ * Z;

    // Linear counting (same as in baseline file)
    std::size_t V = std::count(registers_.begin(), registers_.end(),
                               static_cast<std::uint8_t>(0));
    if (V != 0) {
      double lc =
          m_ * std::log(static_cast<double>(m_) / static_cast<double>(V));
      if (lc <= 5.0 * m_)
        return lc;
    }
    return raw;
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

// ---------------- Packed 6-bit registers ----------------
// Store m registers, each in [0..63] (6 bits). For our HLL with 32-bit hash,
// maximum rho is (32 - B + 1), so 6 bits is more than enough for typical B>=4.

class PackedRegisters6 {
public:
  explicit PackedRegisters6(std::size_t m)
      : m_(m), bits_per_(6), data_((m_ * bits_per_ + 63) / 64, 0ULL) {}

  std::size_t size() const { return m_; }

  std::uint8_t get(std::size_t i) const {
    const std::size_t bitpos = i * bits_per_;
    const std::size_t word = bitpos / 64;
    const std::size_t off = bitpos % 64;

    if (off <= 58) {
      std::uint64_t v = data_[word] >> off;
      return static_cast<std::uint8_t>(v & 0x3FULL);
    } else {
      // crosses boundary
      const std::size_t low_bits = 64 - off;
      std::uint64_t low = data_[word] >> off;
      std::uint64_t high = data_[word + 1] & ((1ULL << (6 - low_bits)) - 1ULL);
      std::uint64_t v = (high << low_bits) | low;
      return static_cast<std::uint8_t>(v & 0x3FULL);
    }
  }

  void set_max(std::size_t i, std::uint8_t value) {
    value &= 0x3F; // keep 6 bits
    std::uint8_t cur = get(i);
    if (value <= cur)
      return;
    set(i, value);
  }

  std::size_t bytes_used() const {
    return data_.size() * sizeof(std::uint64_t);
  }

private:
  void set(std::size_t i, std::uint8_t value) {
    const std::size_t bitpos = i * bits_per_;
    const std::size_t word = bitpos / 64;
    const std::size_t off = bitpos % 64;

    if (off <= 58) {
      std::uint64_t mask = 0x3FULL << off;
      data_[word] =
          (data_[word] & ~mask) | (static_cast<std::uint64_t>(value) << off);
    } else {
      // crosses boundary
      const std::size_t low_bits = 64 - off;
      const std::size_t high_bits = 6 - low_bits;

      std::uint64_t low_mask = ((1ULL << low_bits) - 1ULL) << off;
      data_[word] =
          (data_[word] & ~low_mask) |
          ((static_cast<std::uint64_t>(value) & ((1ULL << low_bits) - 1ULL))
           << off);

      std::uint64_t high_mask = (1ULL << high_bits) - 1ULL;
      data_[word + 1] =
          (data_[word + 1] & ~high_mask) |
          ((static_cast<std::uint64_t>(value) >> low_bits) & high_mask);
    }
  }

  std::size_t m_;
  const std::size_t bits_per_;
  std::vector<std::uint64_t> data_;
};

// ---------------- HyperLogLog improved (packed registers) ----------------

class HyperLogLogPacked6 {
public:
  explicit HyperLogLogPacked6(unsigned B_bits, const HashFuncGen &h)
      : B_(B_bits), m_(1u << B_bits), regs_(m_), hash_(h) {}

  void add(const std::string &value) {
    std::uint32_t x = hash_(value);

    std::uint32_t idx = x >> (32u - B_);
    std::uint32_t w = x << B_;

    std::uint8_t rho;
    if (w != 0)
      rho = static_cast<std::uint8_t>(__builtin_clz(w) + 1);
    else
      rho = static_cast<std::uint8_t>((32u - B_) + 1u);

    regs_.set_max(idx, rho);
  }

  double estimate() const {
    const double alpha_m = alpha_for_m(m_);

    double sum = 0.0;
    std::size_t V = 0;
    for (std::size_t i = 0; i < m_; ++i) {
      std::uint8_t r = regs_.get(i);
      sum += std::ldexp(1.0, -static_cast<int>(r));
      if (r == 0)
        ++V;
    }

    double Z = 1.0 / sum;
    double raw = alpha_m * m_ * m_ * Z;

    // Same linear counting rule as baseline (for a fair comparison)
    if (V != 0) {
      double lc =
          m_ * std::log(static_cast<double>(m_) / static_cast<double>(V));
      if (lc <= 5.0 * m_)
        return lc;
    }
    return raw;
  }

  std::size_t m() const { return m_; }
  std::size_t bytes_used() const { return regs_.bytes_used(); }

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
  PackedRegisters6 regs_;
  const HashFuncGen &hash_;
};

// ---------------- main experiment (baseline vs improved) ----------------

int main() {
  const unsigned B_bits = 10; // same B as baseline run
  const std::size_t m = 1u << B_bits;

  const std::size_t num_streams = 5;
  const std::size_t stream_length = 100000;
  const std::size_t step_percent = 5;

  if (step_percent == 0 || step_percent > 100 || (100 % step_percent) != 0) {
    std::cerr << "step_percent must divide 100\n";
    return 1;
  }

  const std::size_t num_steps = 100 / step_percent;
  const std::size_t step_size = stream_length / num_steps;

  std::uint64_t base_seed = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());

  std::cout << "Stage 4: HyperLogLog baseline vs packed-6bit registers\n";
  std::cout << "B = " << B_bits << ", m = " << m << "\n";
  std::cout
      << "Theoretical relative std.dev scale ~ 1.04/sqrt(m) and 1.3/sqrt(m)\n";

  // Memory numbers (bytes)
  std::size_t base_bytes =
      m * sizeof(std::uint8_t); // exact for baseline registers
  std::size_t packed_bytes =
      (m * 6 + 7) / 8; // theoretical minimum for 6-bit packing
  std::cout << "Memory (registers only):\n";
  std::cout << "  baseline: " << base_bytes << " bytes (uint8_t * m)\n";
  std::cout << "  packed6 : ~" << packed_bytes << " bytes (6 bits * m)\n";
  std::cout << "  saving  : ~" << (base_bytes - packed_bytes) << " bytes (~"
            << (100.0 * (base_bytes - packed_bytes) / base_bytes) << "%)\n\n";

  // Output files with a prefix to avoid overwriting baseline results
  const std::string prefix = "stage4_";

  std::ofstream csv_all(prefix + "hll_results_all_streams.csv");
  csv_all << "stream_id,step_index,items_processed,true_F0,"
             "hll_estimate_base,hll_estimate_packed6\n";

  for (std::size_t s = 0; s < num_streams; ++s) {
    std::uint64_t seed = base_seed + s * 1000;
    RandomStreamGen stream_gen(seed + 1);
    HashFuncGen hash("hll_salt_" + std::to_string(seed + 2));

    HyperLogLogBase hll_base(B_bits, hash);
    HyperLogLogPacked6 hll_packed(B_bits, hash);

    std::vector<std::string> stream = stream_gen.generate(stream_length);

    std::unordered_set<std::string> distinct;
    distinct.reserve(stream_length * 2);

    std::size_t items_processed = 0;

    std::string filename = prefix + "hll_stream_" + std::to_string(s) + ".csv";
    std::ofstream csv_stream(filename);
    csv_stream << "step_index,items_processed,true_F0,"
                  "hll_estimate_base,hll_estimate_packed6\n";

    for (std::size_t step = 0; step < num_steps; ++step) {
      std::size_t until = (step + 1) * step_size;
      if (until > stream_length)
        until = stream_length;

      for (; items_processed < until; ++items_processed) {
        const std::string &val = stream[items_processed];
        distinct.insert(val);
        hll_base.add(val);
        hll_packed.add(val);
      }

      double true_f0 = static_cast<double>(distinct.size());
      double est_base = hll_base.estimate();
      double est_packed = hll_packed.estimate();

      csv_stream << step << ',' << items_processed << ',' << true_f0 << ','
                 << est_base << ',' << est_packed << '\n';

      csv_all << s << ',' << step << ',' << items_processed << ',' << true_f0
              << ',' << est_base << ',' << est_packed << '\n';
    }
  }

  std::cout << "Done. Generated Stage 4 CSV with prefix '" << prefix << "':\n";
  std::cout << "  " << prefix << "hll_stream_*.csv\n";
  std::cout << "  " << prefix << "hll_results_all_streams.csv\n";
  std::cout << "Next: reuse your Python plotting to compare base vs packed6.\n";
  return 0;
}