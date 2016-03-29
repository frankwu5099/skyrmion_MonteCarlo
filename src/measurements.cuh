#ifndef PARAMS_H
#define PARAMS_H
#include "params.cuh"
#endif
#define coo(k, j, i) ((k) * Nplane + (j) * SpinSize + (i))
#define coo2D(j, i) ((j) * SpinSize + (i))
#define CAL(confx, confy, confz, rng, hs, invT) {flipTLBR_thin(confx, confy, confz, rng, hs, invT);\
  flipBLTR_thin(confx, confy, confz, rng, hs, invT);}

