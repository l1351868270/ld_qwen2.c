
#pragma once

namespace ld_infer {
namespace cuda {
constexpr int WARPGROUP_THREADS = 128;
constexpr int WARP_THREADS = 32;

__device__ bool thread0() {
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
}

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

} // namespace cuda
} // namespace ld_infer