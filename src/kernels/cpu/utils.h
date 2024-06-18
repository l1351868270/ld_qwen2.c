
#pragma once
#include <thread>

namespace ld_infer {

int get_num_cpus() {
    if (std::thread::hardware_concurrency() != 0) {
        return std::thread::hardware_concurrency();
    }
    else {
        return 8;
    }
}

int NUM_CPUS = get_num_cpus();

} // namespace ld_infer