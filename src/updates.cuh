#ifndef PARAMS_H
#define PARAMS_H
#include "params.cuh"
#endif
#define SSF(confx, confy, confz, rng, hs, invT) {flipTLBR_thin<<<grid, block>>>(confx, confy, confz, rng, hs, invT);\
  flipBLTR_thin<<<grid, block>>>(confx, confy, confz, rng, hs, invT);}

__global__ void flipTLBR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT);
__global__ void flipBLTR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT);
__global__ void flipTLBR_2D(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT);
__global__ void flipBLTR_2D(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT);
__global__ void flip1_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT);
__global__ void flip2_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT);
__global__ void flip3_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT);
//__global__ void relaxTLBR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float H);
//__global__ void relaxBLTR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float H);
