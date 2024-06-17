
#pragma once

namespace ld_infer {
constexpr int WARPGROUP_THREADS = 128;
constexpr int WARP_THREADS = 32;

__device__ bool thread0() {
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
}

} // namespace ld_infer