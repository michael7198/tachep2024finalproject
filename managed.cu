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

//    if (threadIdx.x == 31){
//        printf("%d,%d: %d %d \n", threadIdx.x, threadIdx.y, temp[lindex_x][lindex_y], in[gindex_y+size*gindex_x]);
//    }

    
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

        
        // Store the result
        out[gindex_y+size*gindex_x] = result;
//        if (result == 8){
//            printf("%d,%d::%d,%d::%d,%d   %d %d %d %d %d    %d %d %d %d %d  \n",gindex_x, gindex_y, lindex_x, lindex_y, threadIdx.x, threadIdx.y,
//                   temp[lindex_x-2][lindex_y],
//                   temp[lindex_x-1][lindex_y],
//                   temp[lindex_x][lindex_y],
//                   temp[lindex_x+1][lindex_y],
//                   temp[lindex_x+2][lindex_y],
//
//                   temp[lindex_x][lindex_y -2],
//                   temp[lindex_x][lindex_y -1],
//                   temp[lindex_x][lindex_y],
//                   temp[lindex_x][lindex_y + 1],
//                   temp[lindex_x][lindex_y + 2]);
//        }
////
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
    
    
    int *dA, *dB, *dAstencil, *dBstencil, *dC;
    int size = (DSIZE)*(DSIZE) * sizeof(int);
    
    cudaMallocManaged((void **)&dA, size);
    cudaMallocManaged((void **)&dB, size);
    cudaMallocManaged((void **)&dAstencil, size);
    cudaMallocManaged((void **)&dBstencil, size);
    cudaMallocManaged((void **)&dC, size);

    //construct matrices
    for (int i=0; i<DSIZE*DSIZE; i++){
        dA[i] = 1;
        dB[i] = 3;
    }

    cudaCheckErrors("ERROR at mallocmanaged");
    
    cudaDeviceSynchronize();
    
    cudaCheckErrors("ERROR at synchronize");

    
	// Launch stencil_2d() kernel on GPU
	int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	// Launch the kernel 
 

    stencil_2d<<<grid,block,(BLOCK_SIZE + 2 * RADIUS)*(BLOCK_SIZE + 2 * RADIUS)*sizeof(int)>>>(dA, dAstencil, DSIZE);
    stencil_2d<<<grid,block,(BLOCK_SIZE + 2 * RADIUS)*(BLOCK_SIZE + 2 * RADIUS)*sizeof(int)>>>(dB, dBstencil, DSIZE);

    
    matrix_mul_gpu<<<grid, block>>>(dAstencil, dBstencil, dC, DSIZE);

    
    cudaCheckErrors("ERROR at kernel");
    
    cudaDeviceSynchronize();
    
    cudaCheckErrors("ERROR at synchronize (post kernel execution)");
    
    //check: I generated the correct answers and chose a few spots on the edges
    //       and center etc to check the c/cuda/etc answers are correct
    
    if (dA[0] != 1){ //check array A is constructed properly aka rand() worked
        std::cout << "Error in generating Array A, value (0,0) should be 1, returned " << dA[0];
        return 1;
    }
    if (dB[0] != 3){ //check array B is constructed properly aka rand() worked
        std::cout << "Error in generating Array B, value (0,0) should be 3, value is " << dB[0];
        return 1;
    }
    if (dAstencil[0] != 1){ //confirm stencil did nothing to corner values on A
        std::cout << "Stencil not applied properly on A, value (0,0) should be 1, returned " << dAstencil[0];
        return 1;
    }
    if (dAstencil[1*DSIZE + 157] != 1){ //confirm stencil did nothing to edge values on A
        std::cout << "Stencil not applied properly on A, value (1,157) should be 1, returned " << dAstencil[1*DSIZE + 150];
        return 1;
    }
    if (dAstencil[342*DSIZE + 58] != 9){ //confirm stencil worked in center on A
        std::cout << "Stencil not applied properly on A, value (342,58) should be 9, returned " << dAstencil[342*DSIZE + 58];
        return 1;
    }
    if (dBstencil[0] != 3){ //confirm stencil did nothing to corner values on B
        std::cout << "Stencil not applied properly on B, value (0,0) should be 3, returned " << dBstencil[0];
        return 1;
    }
    if (dBstencil[1*DSIZE + 157] != 3){ //confirm stencil did nothing to edge values on B
        std::cout << "Stencil not applied properly on B, value (1,157) should be 3, returned " << dBstencil[1*DSIZE + 150];
        return 1;
    }
    if (dBstencil[342*DSIZE + 58] != 27){ //confirm stencil worked in center on B
        std::cout << "Stencil not applied properly on B, value (342,58) should be 27, returned " << dBstencil[342*DSIZE + 58];
        return 1;
    }
    if (dC[0] != 1536){ //confirm matrix multiplication worked on corner
        std::cout << "Matrix multiplication incorrect, value (0,0) should be 1536, returned " << dC[0];
        return 1;
    }
    if (dC[1*DSIZE + 157] != 13728){ //confirm matrix multiplication worked on edge
        std::cout << "Matrix multiplication incorrect, value (1,157) should be 13728, returned " << dC[1*DSIZE + 157];
        return 1;
    }
    if (dC[235*DSIZE+461] != 123456){ //confirm matrix multiplication worked in center
        std::cout << "Matrix multiplication incorrect, value (235,461) should be 123456, returned " << dC[35*DSIZE+61];

        return 1;
    }


    
    
	// Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dAstencil);
    cudaFree(dBstencil);
    cudaFree(dC);
	printf("Success!\n ");
    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t0))/CLOCKS_PER_SEC;
    printf("Compute took %f seconds\n", t3sum);



	return 0;
}


