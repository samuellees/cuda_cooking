# cuda_matrix_multiply
Learning CUDA: Implement and optimize matrix multiply

### Feature
This is the shared_memory version. To simplify the logic, the size of block and shared memory in GPU is 4x4, so the shape of A, B and C should be multiples of 4.