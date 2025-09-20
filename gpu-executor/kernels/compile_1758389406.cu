// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: OpenCL
// Timestamp: 1758389406
// Type: compile_only

__kernel void matrixMultiply(__global const float *A,
                             __global const float *B,
                             __global float *C,
                             int A_rows,
                             int A_cols,
                             int B_cols) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row < A_rows && col < B_cols) {
    float sum = 0.0f;
    for (int k = 0; k < A_cols; ++k) {
      sum += A[row * A_cols + k] * B[k * B_cols + col];
    }
    C[row * B_cols + col] = sum;
  }
}


//Optimized version utilizing workgroup level shared memory for better cache utilization.

__kernel void matrixMultiplyShared(__global const float *A,
                                   __global const float *B,
                                   __global float *C,
                                   int A_rows,
                                   int A_cols,
                                   int B_cols) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    __local float sharedA[TILE_SIZE][TILE_SIZE];
    __local float sharedB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    const int numTiles = (A_cols + TILE_SIZE -1) / TILE_SIZE;

    if (row < A_rows && col < B_cols) {
        for (int tile = 0; tile < numTiles; ++tile) {
            int globalRowA = row;
            int globalColA = tile * TILE_SIZE + localCol;
            int globalRowB = tile * TILE_SIZE + localRow;
            int globalColB = col;

            sharedA[localRow][localCol] = (globalColA < A_cols && globalRowA < A_rows) ? A[globalRowA * A_cols + globalColA] : 0.0f;
            sharedB[localRow][localCol] = (globalRowB < A_cols && globalColB < B_cols) ? B[globalRowB * B_cols + globalColB] : 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += sharedA[localRow][k] * sharedB[k][localCol];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
        C[row * B_cols + col] = sum;
    }
}

//Define TILE_SIZE appropriately for your hardware.  Experimentation is key.  A power of 2 is generally recommended.
#define TILE_SIZE 32