// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#ifndef ALL_REDUCE_H
#define ALL_REDUCE_H

#include <mpi.h>

void recursiveHalvingDoublingAllReduceGPU(float* output, 
                                  const float* input, 
                                  int64_t total_elems, 
                                  float* buf,  
                                  float* recv_buf,  
                                  float* intermediate_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);

void ringAllReduceGPU(float* output, 
                    const float* input, 
                    int64_t total_elems, 
                    float* intermediate_buf,        // Input size / world size
                    float* d_buf,                   // Input size
                    float* d_send,                  // Input size / world size
                    float* d_tmp,                   // Input size / world size
                    MPI_Comm comm = MPI_COMM_WORLD);

#endif // ALL_REDUCE_H
