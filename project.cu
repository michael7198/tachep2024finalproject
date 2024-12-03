#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <time.h>


#define N 512
#define DSIZE 512
#define RADIUS 2

#define BLOCK_SIZE 32

// error checking macro
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)


__global__ void stencil_2d(int *in, int *out, int size) {
    
    int gindex_x = blockIdx.x*blockDim.x + threadIdx.x;
    int gindex_y = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
    int lindex_x = threadIdx.x + RADIUS;
    int lindex_y = threadIdx.y + RADIUS;

    // Read input elements into shared memory
    temp[lindex_x][lindex_y] = in[gindex_y+size*gindex_x];


    
    if ((threadIdx.x < RADIUS)&&(gindex_x >= RADIUS)){
        temp[lindex_x - RADIUS][lindex_y] = in[gindex_y + (size * (gindex_x - RADIUS))];
        temp[lindex_x + BLOCK_SIZE][lindex_y] = in[gindex_y + size*(gindex_x + BLOCK_SIZE)];
    }
    if ((threadIdx.y < RADIUS)&&(gindex_x >= RADIUS)){
        temp[lindex_x][lindex_y - RADIUS] = in[gindex_y - RADIUS + size*gindex_x];
        temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_y + BLOCK_SIZE + size*gindex_x];
    }
    
    __syncthreads();

    
    if ((std::abs((gindex_x+RADIUS)%(size-1)-RADIUS) >= RADIUS) && (std::abs((gindex_y+RADIUS)%(size-1)-RADIUS) >= RADIUS)){


        // Apply the stencil
        int result = -1*temp[lindex_x][lindex_y];
        
        
        for (int offset = -1*RADIUS; offset <= RADIUS; offset++){
            result += temp[lindex_x + offset][lindex_y];
            result += temp[lindex_x][lindex_y + offset];
        }
    
    __syncthreads();


        // Store the result
        out[gindex_y+size*gindex_x] = result;
        
    }
    else{
        out[gindex_y +size*gindex_x] = in[gindex_y +size*gindex_x];
    }
        
    }

    // Square matrix multiplication on GPU : C = A * B
    __global__ void matrix_mul_gpu(const int *A, const int *B, int *C, int size) {

        // create thread x index
        // create thread y index
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;
        // Make sure we are not out of range
        if ((idx < size) && (idy < size)) {
            float temp = 0;
            for (int i = 0; i < size; i++){
                temp += (A[size*idx + i]*B[idy + size*i]);
            }
            C[idy*size+idx] = temp;
        }

}


int main(void) {

    double t0, t3, t3sum;


    t0 = clock();

    //device copies of the intermediate stencil results arent necessary
    //but they will be good to have for debugging
    
    
    int *A, *B, *Astencil, *Bstencil, *C;
    int *dA, *dB, *dAstencil, *dBstencil, *dC;
    int size = (N)*(N) * sizeof(int);
    
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    Astencil = (int *)malloc(size);
    Bstencil = (int *)malloc(size);
    C = (int *)malloc(size);

    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dAstencil, size);
    cudaMalloc((void **)&dBstencil, size);
    cudaMalloc((void **)&dC, size);

    //construct matrices
    for (int i=0; i<DSIZE*DSIZE; i++){
        A[i] = 1;
        B[i] = 3;
    }

    cudaCheckErrors("ERROR at malloc");
    
	// Copy to device
	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    
    cudaCheckErrors("ERROR at copy to device");

    
	// Launch stencil_2d() kernel on GPU
	int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	// Launch the kernel 
	// Properly set memory address for first element on which the stencil will be applied

    stencil_2d<<<grid,block>>>(dA, dAstencil, DSIZE);
    stencil_2d<<<grid,block>>>(dB, dBstencil, DSIZE);

    matrix_mul_gpu<<<grid, block>>>(dAstencil, dBstencil, dC, DSIZE);

    cudaCheckErrors("ERROR at kernel");
    
    
    // Copy result back to host
    cudaMemcpy(Astencil, dAstencil, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Bstencil, dBstencil, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);


    cudaCheckErrors("ERROR at copy back");

    //check: I generated the correct answers and chose a few spots on the edges
    //       and center etc to check the c/cuda/etc answers are correct
    
    if (A[0] != 1){ //check array A is constructed properly aka rand() worked
        std::cout << "Error in generating Array A, value (0,0) should be 1, returned " << A[0];
        return 1;
    }
    if (B[0] != 3){ //check array B is constructed properly aka rand() worked
        std::cout << "Error in generating Array B, value (0,0) should be 3, value is " << B[0];
        return 1;
    }
    if (Astencil[0] != 1){ //confirm stencil did nothing to corner values on A
        std::cout << "Stencil not applied properly on A, value (0,0) should be 1, returned " << Astencil[0];
        return 1;
    }
    if (Astencil[1*DSIZE + 157] != 1){ //confirm stencil did nothing to edge values on A
        std::cout << "Stencil not applied properly on A, value (1,157) should be 1, returned " << Astencil[1*DSIZE + 150];
        return 1;
    }
    if (Astencil[342*DSIZE + 58] != 9){ //confirm stencil worked in center on A
        std::cout << "Stencil not applied properly on A, value (342,58) should be 9, returned " << Astencil[342*DSIZE + 58];
        return 1;
    }
    if (Bstencil[0] != 3){ //confirm stencil did nothing to corner values on B
        std::cout << "Stencil not applied properly on B, value (0,0) should be 3, returned " << Bstencil[0];
        return 1;
    }
    if (Bstencil[1*DSIZE + 157] != 3){ //confirm stencil did nothing to edge values on B
        std::cout << "Stencil not applied properly on B, value (1,157) should be 3, returned " << Bstencil[1*DSIZE + 150];
        return 1;
    }
    if (Bstencil[342*DSIZE + 58] != 27){ //confirm stencil worked in center on B
        std::cout << "Stencil not applied properly on B, value (342,58) should be 27, returned " << Bstencil[342*DSIZE + 58];
        return 1;
    }
    if (C[0] != 1536){ //confirm matrix multiplication worked on corner
        std::cout << "Matrix multiplication incorrect, value (0,0) should be 1536, returned " << C[0];
        return 1;
    }
    if (C[1*DSIZE + 157] != 13728){ //confirm matrix multiplication worked on edge
            std::cout << "Matrix multiplication incorrect, value (1,157) should be 13728, returned " << C[1*DSIZE + 157];
            return 1;
    }
    if (C[235*DSIZE+461] != 123456){ //confirm matrix multiplication worked in center
        std::cout << "Matrix multiplication incorrect, value (235,461) should be 123456, returned " << C[35*DSIZE+61];
        return 1;
    }


	// Cleanup
    free(A);
    free(B);
    free(Astencil);
    free(Bstencil);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dAstencil);
    cudaFree(dBstencil);
    cudaFree(dC);
	printf("Success!\n");
    
    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t0))/CLOCKS_PER_SEC;
    printf("Compute took %f seconds\n", t3sum);



	return 0;
}


