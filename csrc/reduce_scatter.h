#ifndef REDUCE_SCATTER_H
#define REDUCE_SCATTER_H

#include <mpi.h>

void recursiveHalvingReduceScatterGPU(float* output, 
    const float* input, 
    int total_elems,
    float* buf, 
    float* recv_buf, 
    MPI_Comm comm = MPI_COMM_WORLD);

void ringReduceScatterGPU(float* output, 
    const float* input, 
    int total_elems, 
    float* d_buf, 
    float* d_send, 
    float* d_tmp, 
    MPI_Comm comm = MPI_COMM_WORLD);

#endif // REDUCE_SCATTER_H
