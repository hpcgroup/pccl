// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include <cassert>
#include <cmath>

#include "all_gather.h"
#include "all_reduce.h"
#include "common.h"
#include "reduce_scatter.h"


// Performs an all-reduce on GPU tensors via recursive-halving reduce-scatter followed by recursive-doubling all-gather.
//  - output: CUDA device pointer where the final gathered tensor will be stored.
//  - input: CUDA device pointer to the local block of size block_size.
//  - total_elems: total number of elements in output (P * block_size).
//  - buf: main working buffer for the algorithm
//  - recv_buf: buffer to receive data before it is processed
//  - comm: MPI communicator (default MPI_COMM_WORLD).
void recursiveHalvingDoublingAllReduceGPU(float* output,
                                const float* input,
                                int64_t total_elems,
                                float* buf,                 // Same as input size
                                float* recv_buf,            // Same as input size
                                float* intermediate_buf,    // Input size / world size
                                MPI_Comm comm) {
    recursiveHalvingReduceScatterGPU(intermediate_buf, input, total_elems, buf, recv_buf, comm);

    // allgather uses void* so multiply total_elems by size of float dtype
    recursiveDoublingAllGatherGPU(output, intermediate_buf, total_elems*sizeof(float), recv_buf, comm);
}

// Performs an all-reduce on GPU tensors via ring reduce-scatter followed by ring all-gather.
void ringAllReduceGPU(float* output,
                    const float* input,
                    int64_t total_elems,
                    float* intermediate_buf,        // Input size / world size
                    float* d_buf,                   // Input size
                    float* d_send,                  // Input size / world size
                    float* d_tmp,                   // Input size / world size
                    MPI_Comm comm) {
    ringReduceScatterGPU(intermediate_buf, input, total_elems, d_buf, d_send, d_tmp);

    // allgather uses void* so multiply total_elems by size of float dtype
    ringAllGatherGPU(output, intermediate_buf, total_elems*sizeof(float), comm);
}

