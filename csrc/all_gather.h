// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#ifndef ALL_GATHER_H
#define ALL_GATHER_H

#include <mpi.h>

void recursiveDoublingAllGatherGPU(void* output, 
                                  const void* input, 
                                  int total_elems, 
                                  void* recv_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);

#endif // ALL_GATHER_H
