// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#ifndef REDUCE_SCATTER_H
#define REDUCE_SCATTER_H

#include <mpi.h>

void recursiveHalvingReduceScatterGPU(float* output, 
    const float* input, 
    int64_t total_elems,
    float* buf, 
    float* recv_buf, 
    MPI_Comm comm = MPI_COMM_WORLD);

void ringReduceScatterGPU(float* output, 
    const float* input, 
    int64_t total_elems, 
    float* d_buf, 
    float* d_send, 
    float* d_tmp, 
    MPI_Comm comm = MPI_COMM_WORLD);

#endif // REDUCE_SCATTER_H
