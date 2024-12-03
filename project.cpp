#include <stdlib.h>
#include <iostream>
#include <time.h>

#define DSIZE 512
#define RADIUS 2


float dotprod(int* A, int* B, int size){
    float sum = 0;
    for (int i=0; i<size; i++){
        sum += (A[i])*(B[i]);
    }
    return sum;
}

void matmult(int* A, int* B, int* result, int nrows, int ncols){
    for (int i=0; i<ncols; i++){
        int v1[nrows];
        for (int j=0; j<nrows; j++){
            //first construct column
            v1[j] = B[j*ncols + i];

        }
        for (int j=0; j<nrows; j++){
            //construct row vector
            int v2[ncols];
            for (int k=0; k<ncols; k++){
                v2[k] = A[j*ncols+k];
            }
            //now do matrix multiplication
            result[j*ncols+i] = dotprod(v1,v2,sizeof(v1)/sizeof(int));
        }
    }
}

void stencil(int* in, int* out, int radius, int size){
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
            int result = 0;
            if ((std::abs((i+radius)%(size-1)-radius) >= radius) && (std::abs((j+radius)%(size-1)-radius) >= radius)){
                for (int k=-1*radius; k<=radius; k++){
                    result += in[(j + k)*size + i]; //vertical
                    result += in[j * size + (i + k)]; //horizontal
                }
                result -= in[j*size + i];
                out[j*size + i] = result;
            }
            else {
                out[j*size + i] = in[j*size + i];
            }
        }
    }
}

/*
 A: construct 2 square matrices, A and B (size >= 512) with integer values
 B: run stencil on A and B (radius >= 2)
 C: multiply A*B
 D: check results by writing tests
     -check corners, edges, middle for stencil
     -check matrix mult
     -not sure how to do this quickly but shouldnt be too hard?
*/


int main(){
    
    double t0, t3, t3sum;

    t0 = clock();

    int A[DSIZE*DSIZE];
    int B[DSIZE*DSIZE];
    int Astencil[DSIZE*DSIZE];
    int Bstencil[DSIZE*DSIZE];
    int C[DSIZE*DSIZE];
    
    for (int i=0; i<DSIZE*DSIZE; i++){
        A[i] = 1;
        B[i] = 3;
    }
    /*
     code to make text file of arrays A and B, to check values
     
    //print elements of A
    for (int i=0; i<DSIZE*DSIZE; i++){
        std::cout << A[i] << " ";
    }
    std::cout << "\n";
    //print elements of B
    for (int i=0; i<DSIZE*DSIZE; i++){
        std::cout << B[i] << " ";
    }
     */
    

    stencil(A, Astencil, RADIUS, DSIZE);
    stencil(B, Bstencil, RADIUS, DSIZE);
    
    matmult(Astencil, Bstencil, C, DSIZE, DSIZE);
    

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
        std::cout << "Matrix multiplication incorrect, value (157,0) should be 13728, returned " << C[1*DSIZE + 157];
        return 1;
    }
    if (C[235*DSIZE+461] != 123456){ //confirm matrix multiplication worked in center
        std::cout << "Matrix multiplication incorrect, value (235,461) should be 123456, returned " << C[35*DSIZE+61];
        return 1;
    }
    
    std::cout << "Success!\n";
    
    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t0))/CLOCKS_PER_SEC;
    printf("Compute took %f seconds\n", t3sum);

}


