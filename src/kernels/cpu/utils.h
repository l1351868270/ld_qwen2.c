
#pragma once
#include <thread>

namespace ld_infer {
namespace cpu {
namespace utils {
int get_num_cpus() {
    const char* path = std::getenv("OMP_NUM_THREADS");
    if (path != nullptr) {
        return std::stoi(path);
    } else if (std::thread::hardware_concurrency() != 0) {
        return std::thread::hardware_concurrency();
    }
    else {
        return 8;
    }
}

int NUM_CPUS = get_num_cpus();
} // namespace utils
} // namespace cpu
} // namespace ld_infer