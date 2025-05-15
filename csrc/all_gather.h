#ifndef ALL_GATHER_H
#define ALL_GATHER_H

#include <mpi.h>

void recursiveDoublingAllGatherGPU(void* output, 
                                  const void* input, 
                                  int total_elems, 
                                  void* recv_buf,  
                                  MPI_Comm comm = MPI_COMM_WORLD);

#endif // ALL_GATHER_H
