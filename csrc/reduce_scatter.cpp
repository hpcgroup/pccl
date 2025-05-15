// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include <cassert>
#include <cmath>

#include "reduce_scatter.h"
#include "common.h"


// Performs a recursive-halving reduce-scatter on GPU tensors.
//  - output: CUDA device pointer where the reduced block (of block_size elements) will be stored.
//  - input: CUDA device pointer to a contiguous buffer of (P * block_size) elements.
//  - total_elems: total number of elements (must equal P * block_size).
//  - comm: MPI communicator (default MPI_COMM_WORLD).
// The reduction operation is elementwise addition.
void recursiveHalvingReduceScatterGPU(float* output, 
    const float* input, 
    int total_elems,
    float* buf, // same as size of input 
    float* recv_buf, // same as size of input 
    MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;
    auto stream = at::cuda::getCurrentCUDAStream();

    // copy the input into buf
    CUDA_CHECK(cudaMemcpyAsync(buf, 
        input, 
        total_elems * sizeof(float), 
        cudaMemcpyDeviceToDevice,
        stream));

    
    // Number of rounds = log2(size); assumes size is a power of 2.
    int rounds = static_cast<int>(std::log2(size));
    int current_blocks = size;
    // curr_ptr points to the beginning of the "active" region within buf.
    float* curr_ptr = buf;

    cudaEvent_t stream_sync_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    for (int r = 0; r < rounds; ++r) {
        // In round r, the group size is:
        int group_size = size / (1 << r);
        // The current buffer holds 'current_blocks' contiguous blocks.
        int half = current_blocks / 2;
        // Number of elements to send/receive in this round.
        int count = half * block_size;

        if ((rank % group_size) < (group_size / 2)) {
            // Lower half: keep the lower half and send the upper half.
            int partner = (rank + half) % size;
            float* send_ptr = curr_ptr + half * block_size;
            // Record an event on the cuda stream.
            CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
            // Wait for the copy to complete.
            CUDA_CHECK(cudaEventSynchronize(stream_sync_event));

            MPI_Sendrecv(send_ptr, count, MPI_FLOAT, partner, r,
                         recv_buf, count, MPI_FLOAT, partner, r,
                         comm, MPI_STATUS_IGNORE);
            // Reduce the received data into the lower half.
            vectorAdd(curr_ptr, recv_buf, count, stream);
            current_blocks = half;
        } else {
            // Upper half: keep the upper half and send the lower half.
            int partner = (rank - half + size) % size;
            float* send_ptr = curr_ptr;  // lower half to be sent.
            // Record an event on the cuda stream.
            CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
            // Wait for the copy to complete.
            CUDA_CHECK(cudaEventSynchronize(stream_sync_event));
            MPI_Sendrecv(send_ptr, count, MPI_FLOAT, partner, r,
                         recv_buf, count, MPI_FLOAT, partner, r,
                         comm, MPI_STATUS_IGNORE);
            float* kept_ptr = curr_ptr + half * block_size;
            vectorAdd(kept_ptr, recv_buf, count, stream);
            curr_ptr = kept_ptr;
            current_blocks = half;
        }
    }

    // After log2(size) rounds, curr_ptr points to a single block (of block_size elements).
    CUDA_CHECK(cudaMemcpyAsync(output, 
        curr_ptr, 
        block_size * sizeof(float), 
        cudaMemcpyDeviceToDevice, 
        stream));

    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}

void ringReduceScatterGPU(float* output, 
    const float* input, 
    int total_elems, 
    float* d_buf, // same as size of input
    float* d_send, // same as size of output
    float* d_tmp, // same as size of output
    MPI_Comm comm
) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    assert(total_elems % size == 0 && "Input tensor size must be divisible by number of processes");
    int block_size = total_elems / size;

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t stream_sync_event;
    
    CUDA_CHECK(cudaMemcpyAsync(d_buf, 
        input, 
        total_elems * sizeof(float), 
        cudaMemcpyDeviceToDevice,
        stream));

    CUDA_CHECK(cudaEventCreateWithFlags(&stream_sync_event, cudaEventDisableTiming));

    for (int step = 0; step < size - 1; step++) {
        // Compute block indices.
        int send_idx = (rank - step - 1 + size) % size;
        int recv_idx = (rank - step - 2 + size) % size;
        int dest     = (rank + 1) % size;
        int source   = (rank - 1 + size) % size;

        // Record an event on the cuda stream.
        CUDA_CHECK(cudaEventRecord(stream_sync_event, stream));
        // Wait for the copy to complete.
        CUDA_CHECK(cudaEventSynchronize(stream_sync_event));

        // Send the block to the right neighbor and receive from the left neighbor.
        MPI_Sendrecv(d_buf + send_idx * block_size, block_size, MPI_FLOAT, dest, 0,
                    d_tmp,  block_size, MPI_FLOAT, source, 0,
                    comm, MPI_STATUS_IGNORE);

        // Reduce the received block into the block at recv_idx.
        vectorAdd(d_buf + recv_idx * block_size, d_tmp, block_size, stream);
    }
     
    CUDA_CHECK(cudaMemcpyAsync(output, 
        d_buf + rank * block_size, 
        block_size * sizeof(float), 
        cudaMemcpyDeviceToDevice,
        stream));

    CUDA_CHECK(cudaEventDestroy(stream_sync_event));
}

