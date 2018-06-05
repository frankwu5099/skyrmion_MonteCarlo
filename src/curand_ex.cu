/*
 * This program uses the device CURAND API to calculate what 
 * proportion of pseudo-random ints have low bit set.
 * It then generates uniform results to calculate how many
 * are greater than .5.
 * It then generates  normal results to calculate how many 
 * are within one standard deviation of the mean.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}


__global__ void generate_kernel(curandState *state, 
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int n = 0; n < 10000; n++) {
        x = curand(&localState);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(curandState *state, 
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    float x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int n = 0; n < 10; n++) {
        x = curand_uniform(&localState);
        /* Check if > .5 */        
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}


int main(int argc, char *argv[])
{
    int i;
    unsigned int total;
    curandState *devStates;
    curandStateMRG32k3a *devMRGStates;
    unsigned int *devResults, *hostResults;
    bool useMRG = 0;
    bool doubleSupported = 0;
    int device;
    struct cudaDeviceProp properties;  

    /* check for double precision support */
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties,device));
    if ( properties.major >= 2 || (properties.major == 1 && properties.minor >= 3) ) {
        doubleSupported = 1;
    }
    
    /* Check for MRG32k3a option (default is XORWOW) */
    if ((argc == 2) && (strcmp(argv[1],"-m") == 0)) {
        useMRG = 1;
        if (!doubleSupported){
            printf("MRG32k3a requires double precision\n");
            printf("^^^^ test WAIVED due to lack of double precision\n");
            return EXIT_SUCCESS;
        }
    }        

    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(64 * 64, sizeof(int));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, 64 * 64 * sizeof(unsigned int)));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * sizeof(unsigned int)));

    /* Allocate space for prng states on device */
    if (!useMRG) {
        CUDA_CALL(cudaMalloc((void **)&devStates, 64 * 64 * sizeof(curandState)));
    } else {
        CUDA_CALL(cudaMalloc((void **)&devMRGStates, 64 * 64 * sizeof(curandStateMRG32k3a)));
    }
    
    /* Setup prng states */
    if (!useMRG) {
        setup_kernel<<<64, 64>>>(devStates);
    } else {
        setup_kernel<<<64, 64>>>(devMRGStates);
    }
    
    /* Generate and use pseudo-random  */
    for(i = 0; i < 50; i++) {
        if (!useMRG) {
            generate_kernel<<<64, 64>>>(devStates, devResults);
        } else {
            generate_kernel<<<64, 64>>>(devMRGStates, devResults);
        }
    }
    
    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
        sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    printf("Fraction with low bit set was %10.13f\n", 
        (float)total / (64.0f * 64.0f * 10000.0f * 50.0f));
        
    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
              sizeof(unsigned int)));

    /* Generate and use uniform pseudo-random  */
    for(i = 0; i < 50; i++) {
        if (!useMRG) {
            generate_uniform_kernel<<<64, 64>>>(devStates, devResults);
        } else {
            generate_uniform_kernel<<<64, 64>>>(devMRGStates, devResults);
        }
    }

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
        sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    printf("Fraction of uniforms > 0.5 was %10.13f\n", 
        (float)total / (64.0f * 64.0f * 10000.0f * 50.0f));
    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
              sizeof(unsigned int)));


    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
        sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    printf("Fraction of normals within 1 standard deviation was %10.13f\n", 
        (float)total / (64.0f * 64.0f * 10000.0f * 50.0f));

    /* Cleanup */
    if (!useMRG) {
        CUDA_CALL(cudaFree(devStates));
    } else {
        CUDA_CALL(cudaFree(devMRGStates));
    }    
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_example PASSED\n");
    return EXIT_SUCCESS;
}
