#ifndef UPDATE_H
#define UPDATE_H
#include "params.cuh"
#ifdef THIN
__global__ void flipTLBR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparamters, float Cparameter);
__global__ void flipBLTR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparamters, float Cparameter);
#endif
#ifdef SQ
__global__ void flipTLBR_2D(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparamters, float Cparameter);
__global__ void flipBLTR_2D(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparamters, float Cparameter);
#endif
#ifdef TRI
__global__ void flip1_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparamters, float Cparameter);
__global__ void flip2_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparamters, float Cparameter);
__global__ void flip3_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparamters, float Cparameter);
#endif
//__global__ void relaxTLBR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float H);
//__global__ void relaxBLTR_thin(float *confx, float *confy, float *confz, unsigned int *rngState, float H);
#endif
