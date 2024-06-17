
#pragma once
#include <thread>

namespace ld_infer {
constexpr int NUM_CPUS = 8;

int get_num_cpus() {
    if (std::thread::hardware_concurrency() != 0) {
        return std::thread::hardware_concurrency();
    }
    else {
        return NUM_CPUS;
    }
}

} // namespace ld_infer