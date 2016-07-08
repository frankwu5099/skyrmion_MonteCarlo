#include "updates.cuh"
#ifdef TRI
__constant__ unsigned int flip_SpinSize;
__constant__ unsigned int flip_SpinSize_z;
__constant__ unsigned int flip_BlockSize_x;
__constant__ unsigned int flip_BlockSize_y;
__constant__ unsigned int flip_GridSize_x;
__constant__ unsigned int flip_GridSize_y;
__constant__ unsigned int flip_BN;
__constant__ float flip_A; //(0.0)
void move_params_device_flip(){
  cudaMemcpyToSymbol( flip_SpinSize, &H_SpinSize, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_SpinSize_z, &H_SpinSize_z, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_BlockSize_x, &H_BlockSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_BlockSize_y, &H_BlockSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_GridSize_x, &H_GridSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_GridSize_y, &H_GridSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_BN, &H_BN, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_A , &H_A , sizeof(float));
}
__global__ void flip1_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
  //Energy variables
  //extern __shared__ unsigned rngShmem[];
  __shared__ unsigned rngShmem[1024];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float Pparameter = Pparameters[blockIdx.x / flip_BN];
  unsigned int r;
  float du;	//-dE
  float sx, sy, sz;
  float th,phi;
  float hx, hy, hz;
  //float norm;
  const int x = threadIdx.x % (flip_BlockSize_x);
  const int y = (threadIdx.x / flip_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % flip_BN) % flip_GridSize_x) * flip_BlockSize_x + x);
  const int ty =(blockIdx.x / flip_BN) * flip_SpinSize +  3 * ((((blockIdx.x % flip_BN) / flip_GridSize_x) % flip_GridSize_y) * flip_BlockSize_y + y);
  int i, j, ib, jb;
  //0..
  //...
  //...
  i = tx;
  j = ty;
  ib = (i + flip_SpinSize - 1) % flip_SpinSize;
  if((j % flip_SpinSize) == 0)	jb = j + flip_SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  hx = BXPxx * confx[flip_coo2D(j, i+1)] + BYPxx * confx[flip_coo2D(j+1, i)] + BWPxx * confx[flip_coo2D(j+1, i+1)] + BXMxx * confx[flip_coo2D(j, ib)] + BYMxx * confx[flip_coo2D(jb, i)] + BWMxx * confx[flip_coo2D(jb, ib)]\
     + BXPxy * confy[flip_coo2D(j, i+1)] + BYPxy * confy[flip_coo2D(j+1, i)] + BWPxy * confy[flip_coo2D(j+1, i+1)] + BXMxy * confy[flip_coo2D(j, ib)] + BYMxy * confy[flip_coo2D(jb, i)] + BWMxy * confy[flip_coo2D(jb, ib)]\
     + BXPxz * confz[flip_coo2D(j, i+1)] + BYPxz * confz[flip_coo2D(j+1, i)] + BWPxz * confz[flip_coo2D(j+1, i+1)] + BXMxz * confz[flip_coo2D(j, ib)] + BYMxz * confz[flip_coo2D(jb, i)] + BWMxz * confz[flip_coo2D(jb, ib)];
  hy = BXPyx * confx[flip_coo2D(j, i+1)] + BYPyx * confx[flip_coo2D(j+1, i)] + BWPyx * confx[flip_coo2D(j+1, i+1)] + BXMyx * confx[flip_coo2D(j, ib)] + BYMyx * confx[flip_coo2D(jb, i)] + BWMyx * confx[flip_coo2D(jb, ib)]\
     + BXPyy * confy[flip_coo2D(j, i+1)] + BYPyy * confy[flip_coo2D(j+1, i)] + BWPyy * confy[flip_coo2D(j+1, i+1)] + BXMyy * confy[flip_coo2D(j, ib)] + BYMyy * confy[flip_coo2D(jb, i)] + BWMyy * confy[flip_coo2D(jb, ib)]\
     + BXPyz * confz[flip_coo2D(j, i+1)] + BYPyz * confz[flip_coo2D(j+1, i)] + BWPyz * confz[flip_coo2D(j+1, i+1)] + BXMyz * confz[flip_coo2D(j, ib)] + BYMyz * confz[flip_coo2D(jb, i)] + BWMyz * confz[flip_coo2D(jb, ib)];
  hz = BXPzx * confx[flip_coo2D(j, i+1)] + BYPzx * confx[flip_coo2D(j+1, i)] + BWPzx * confx[flip_coo2D(j+1, i+1)] + BXMzx * confx[flip_coo2D(j, ib)] + BYMzx * confx[flip_coo2D(jb, i)] + BWMzx * confx[flip_coo2D(jb, ib)]\
     + BXPzy * confy[flip_coo2D(j, i+1)] + BYPzy * confy[flip_coo2D(j+1, i)] + BWPzy * confy[flip_coo2D(j+1, i+1)] + BXMzy * confy[flip_coo2D(j, ib)] + BYMzy * confy[flip_coo2D(jb, i)] + BWMzy * confy[flip_coo2D(jb, ib)]\
     + BXPzz * confz[flip_coo2D(j, i+1)] + BYPzz * confz[flip_coo2D(j+1, i)] + BWPzz * confz[flip_coo2D(j+1, i+1)] + BXMzz * confz[flip_coo2D(j, ib)] + BYMzz * confz[flip_coo2D(jb, i)] + BWMzz * confz[flip_coo2D(jb, ib)] + H;
  du = - confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //..0
  //...
  i = tx + 2;
  j = ty + 1;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[flip_coo2D(j, ib)] + BYPxx * confx[flip_coo2D(jb, i)] + BWPxx * confx[flip_coo2D(jb, ib)] + BXMxx * confx[flip_coo2D(j, i-1)] + BYMxx * confx[flip_coo2D(j-1, i)] + BWMxx * confx[flip_coo2D(j-1, i-1)]\
     + BXPxy * confy[flip_coo2D(j, ib)] + BYPxy * confy[flip_coo2D(jb, i)] + BWPxy * confy[flip_coo2D(jb, ib)] + BXMxy * confy[flip_coo2D(j, i-1)] + BYMxy * confy[flip_coo2D(j-1, i)] + BWMxy * confy[flip_coo2D(j-1, i-1)]\
     + BXPxz * confz[flip_coo2D(j, ib)] + BYPxz * confz[flip_coo2D(jb, i)] + BWPxz * confz[flip_coo2D(jb, ib)] + BXMxz * confz[flip_coo2D(j, i-1)] + BYMxz * confz[flip_coo2D(j-1, i)] + BWMxz * confz[flip_coo2D(j-1, i-1)];
  hy = BXPyx * confx[flip_coo2D(j, ib)] + BYPyx * confx[flip_coo2D(jb, i)] + BWPyx * confx[flip_coo2D(jb, ib)] + BXMyx * confx[flip_coo2D(j, i-1)] + BYMyx * confx[flip_coo2D(j-1, i)] + BWMyx * confx[flip_coo2D(j-1, i-1)]\
     + BXPyy * confy[flip_coo2D(j, ib)] + BYPyy * confy[flip_coo2D(jb, i)] + BWPyy * confy[flip_coo2D(jb, ib)] + BXMyy * confy[flip_coo2D(j, i-1)] + BYMyy * confy[flip_coo2D(j-1, i)] + BWMyy * confy[flip_coo2D(j-1, i-1)]\
     + BXPyz * confz[flip_coo2D(j, ib)] + BYPyz * confz[flip_coo2D(jb, i)] + BWPyz * confz[flip_coo2D(jb, ib)] + BXMyz * confz[flip_coo2D(j, i-1)] + BYMyz * confz[flip_coo2D(j-1, i)] + BWMyz * confz[flip_coo2D(j-1, i-1)];
  hz = BXPzx * confx[flip_coo2D(j, ib)] + BYPzx * confx[flip_coo2D(jb, i)] + BWPzx * confx[flip_coo2D(jb, ib)] + BXMzx * confx[flip_coo2D(j, i-1)] + BYMzx * confx[flip_coo2D(j-1, i)] + BWMzx * confx[flip_coo2D(j-1, i-1)]\
     + BXPzy * confy[flip_coo2D(j, ib)] + BYPzy * confy[flip_coo2D(jb, i)] + BWPzy * confy[flip_coo2D(jb, ib)] + BXMzy * confy[flip_coo2D(j, i-1)] + BYMzy * confy[flip_coo2D(j-1, i)] + BWMzy * confy[flip_coo2D(j-1, i-1)]\
     + BXPzz * confz[flip_coo2D(j, ib)] + BYPzz * confz[flip_coo2D(jb, i)] + BWPzz * confz[flip_coo2D(jb, ib)] + BXMzz * confz[flip_coo2D(j, i-1)] + BYMzz * confz[flip_coo2D(j-1, i)] + BWMzz * confz[flip_coo2D(j-1, i-1)] + H;
  du = -confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //...
  //.0.
  i = tx + 1;
  j = ty + 2;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[flip_coo2D(j, ib)] + BYPxx * confx[flip_coo2D(jb, i)] + BWPxx * confx[flip_coo2D(jb, ib)] + BXMxx * confx[flip_coo2D(j, i-1)] + BYMxx * confx[flip_coo2D(j-1, i)] + BWMxx * confx[flip_coo2D(j-1, i-1)]\
     + BXPxy * confy[flip_coo2D(j, ib)] + BYPxy * confy[flip_coo2D(jb, i)] + BWPxy * confy[flip_coo2D(jb, ib)] + BXMxy * confy[flip_coo2D(j, i-1)] + BYMxy * confy[flip_coo2D(j-1, i)] + BWMxy * confy[flip_coo2D(j-1, i-1)]\
     + BXPxz * confz[flip_coo2D(j, ib)] + BYPxz * confz[flip_coo2D(jb, i)] + BWPxz * confz[flip_coo2D(jb, ib)] + BXMxz * confz[flip_coo2D(j, i-1)] + BYMxz * confz[flip_coo2D(j-1, i)] + BWMxz * confz[flip_coo2D(j-1, i-1)];
  hy = BXPyx * confx[flip_coo2D(j, ib)] + BYPyx * confx[flip_coo2D(jb, i)] + BWPyx * confx[flip_coo2D(jb, ib)] + BXMyx * confx[flip_coo2D(j, i-1)] + BYMyx * confx[flip_coo2D(j-1, i)] + BWMyx * confx[flip_coo2D(j-1, i-1)]\
     + BXPyy * confy[flip_coo2D(j, ib)] + BYPyy * confy[flip_coo2D(jb, i)] + BWPyy * confy[flip_coo2D(jb, ib)] + BXMyy * confy[flip_coo2D(j, i-1)] + BYMyy * confy[flip_coo2D(j-1, i)] + BWMyy * confy[flip_coo2D(j-1, i-1)]\
     + BXPyz * confz[flip_coo2D(j, ib)] + BYPyz * confz[flip_coo2D(jb, i)] + BWPyz * confz[flip_coo2D(jb, ib)] + BXMyz * confz[flip_coo2D(j, i-1)] + BYMyz * confz[flip_coo2D(j-1, i)] + BWMyz * confz[flip_coo2D(j-1, i-1)];
  hz = BXPzx * confx[flip_coo2D(j, ib)] + BYPzx * confx[flip_coo2D(jb, i)] + BWPzx * confx[flip_coo2D(jb, ib)] + BXMzx * confx[flip_coo2D(j, i-1)] + BYMzx * confx[flip_coo2D(j-1, i)] + BWMzx * confx[flip_coo2D(j-1, i-1)]\
     + BXPzy * confy[flip_coo2D(j, ib)] + BYPzy * confy[flip_coo2D(jb, i)] + BWPzy * confy[flip_coo2D(jb, ib)] + BXMzy * confy[flip_coo2D(j, i-1)] + BYMzy * confy[flip_coo2D(j-1, i)] + BWMzy * confy[flip_coo2D(j-1, i-1)]\
     + BXPzz * confz[flip_coo2D(j, ib)] + BYPzz * confz[flip_coo2D(jb, i)] + BWPzz * confz[flip_coo2D(jb, ib)] + BXMzz * confz[flip_coo2D(j, i-1)] + BYMzz * confz[flip_coo2D(j-1, i)] + BWMzz * confz[flip_coo2D(j-1, i-1)] + H;
  du = -confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}



__global__ void flip2_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
  //Energy variables
  __shared__ unsigned rngShmem[1024];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float Pparameter = Pparameters[blockIdx.x / flip_BN];
  unsigned int r;
  float du;	//-dE
  float sx, sy, sz;
  float th,phi;
  float hx, hy, hz;
  //float norm;
  const int x = threadIdx.x % (flip_BlockSize_x);
  const int y = (threadIdx.x / flip_BlockSize_x);// % flip_BlockSize_y;
  const int tx = 3 * (((blockIdx.x % flip_BN) % flip_GridSize_x) * flip_BlockSize_x + x);
  const int ty = (blockIdx.x / flip_BN) * flip_SpinSize + 3 * (((blockIdx.x % flip_BN) / flip_GridSize_x) * flip_BlockSize_y + y);
  int i, j, ib, jb;
  //----------Spin flip at the bottom and left corner of each thread sqare----------
  i = tx;
  j = ty + 1;
  ib = (i + flip_SpinSize - 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  //...
  //0..
  //...
  hx = BXPxx * confx[flip_coo2D(j, i+1)] + BYPxx * confx[flip_coo2D(jb, i)] + BWPxx * confx[flip_coo2D(jb, i+1)] + BXMxx * confx[flip_coo2D(j, ib)] + BYMxx * confx[flip_coo2D((j-1), i)] + BWMxx * confx[flip_coo2D((j-1), ib)]\
     + BXPxy * confy[flip_coo2D(j, i+1)] + BYPxy * confy[flip_coo2D(jb, i)] + BWPxy * confy[flip_coo2D(jb, i+1)] + BXMxy * confy[flip_coo2D(j, ib)] + BYMxy * confy[flip_coo2D((j-1), i)] + BWMxy * confy[flip_coo2D((j-1), ib)]\
     + BXPxz * confz[flip_coo2D(j, i+1)] + BYPxz * confz[flip_coo2D(jb, i)] + BWPxz * confz[flip_coo2D(jb, i+1)] + BXMxz * confz[flip_coo2D(j, ib)] + BYMxz * confz[flip_coo2D((j-1), i)] + BWMxz * confz[flip_coo2D((j-1), ib)];
  hy = BXPyx * confx[flip_coo2D(j, i+1)] + BYPyx * confx[flip_coo2D(jb, i)] + BWPyx * confx[flip_coo2D(jb, i+1)] + BXMyx * confx[flip_coo2D(j, ib)] + BYMyx * confx[flip_coo2D((j-1), i)] + BWMyx * confx[flip_coo2D((j-1), ib)]\
     + BXPyy * confy[flip_coo2D(j, i+1)] + BYPyy * confy[flip_coo2D(jb, i)] + BWPyy * confy[flip_coo2D(jb, i+1)] + BXMyy * confy[flip_coo2D(j, ib)] + BYMyy * confy[flip_coo2D((j-1), i)] + BWMyy * confy[flip_coo2D((j-1), ib)]\
     + BXPyz * confz[flip_coo2D(j, i+1)] + BYPyz * confz[flip_coo2D(jb, i)] + BWPyz * confz[flip_coo2D(jb, i+1)] + BXMyz * confz[flip_coo2D(j, ib)] + BYMyz * confz[flip_coo2D((j-1), i)] + BWMyz * confz[flip_coo2D((j-1), ib)];
  hz = BXPzx * confx[flip_coo2D(j, i+1)] + BYPzx * confx[flip_coo2D(jb, i)] + BWPzx * confx[flip_coo2D(jb, i+1)] + BXMzx * confx[flip_coo2D(j, ib)] + BYMzx * confx[flip_coo2D((j-1), i)] + BWMzx * confx[flip_coo2D((j-1), ib)]\
     + BXPzy * confy[flip_coo2D(j, i+1)] + BYPzy * confy[flip_coo2D(jb, i)] + BWPzy * confy[flip_coo2D(jb, i+1)] + BXMzy * confy[flip_coo2D(j, ib)] + BYMzy * confy[flip_coo2D((j-1), i)] + BWMzy * confy[flip_coo2D((j-1), ib)]\
     + BXPzz * confz[flip_coo2D(j, i+1)] + BYPzz * confz[flip_coo2D(jb, i)] + BWPzz * confz[flip_coo2D(jb, i+1)] + BWMzz * confz[flip_coo2D(j, ib)] + BYMzz * confz[flip_coo2D((j-1), i)] + BWMzz * confz[flip_coo2D((j-1), ib)] + H;
  du = - confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //----------Spin flip at the top and right corner of each thread sqare----------
  i = tx + 1;
  j = ty;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == 0)	jb = j + flip_SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  //.0.
  //...
  //...
  hx = BXPxx * confx[flip_coo2D(j, ib)] + BYPxx * confx[flip_coo2D((j+1), i)] + BWPxx * confx[flip_coo2D((j+1), ib)] + BXMxx * confx[flip_coo2D(j, i-1)] + BYMxx * confx[flip_coo2D(jb, i)] + BWMxx * confx[flip_coo2D(jb, i-1)]\
     + BXPxy * confy[flip_coo2D(j, ib)] + BYPxy * confy[flip_coo2D((j+1), i)] + BWPxy * confy[flip_coo2D((j+1), ib)] + BXMxy * confy[flip_coo2D(j, i-1)] + BYMxy * confy[flip_coo2D(jb, i)] + BWMxy * confy[flip_coo2D(jb, i-1)]\
     + BXPxz * confz[flip_coo2D(j, ib)] + BYPxz * confz[flip_coo2D((j+1), i)] + BWPxz * confz[flip_coo2D((j+1), ib)] + BXMxz * confz[flip_coo2D(j, i-1)] + BYMxz * confz[flip_coo2D(jb, i)] + BWMxz * confz[flip_coo2D(jb, i-1)];
  hy = BXPyx * confx[flip_coo2D(j, ib)] + BYPyx * confx[flip_coo2D((j+1), i)] + BWPyx * confx[flip_coo2D((j+1), ib)] + BXMyx * confx[flip_coo2D(j, i-1)] + BYMyx * confx[flip_coo2D(jb, i)] + BWMyx * confx[flip_coo2D(jb, i-1)]\
     + BXPyy * confy[flip_coo2D(j, ib)] + BYPyy * confy[flip_coo2D((j+1), i)] + BWPyy * confy[flip_coo2D((j+1), ib)] + BXMyy * confy[flip_coo2D(j, i-1)] + BYMyy * confy[flip_coo2D(jb, i)] + BWMyy * confy[flip_coo2D(jb, i-1)]\
     + BXPyz * confz[flip_coo2D(j, ib)] + BYPyz * confz[flip_coo2D((j+1), i)] + BWPyz * confz[flip_coo2D((j+1), ib)] + BXMyz * confz[flip_coo2D(j, i-1)] + BYMyz * confz[flip_coo2D(jb, i)] + BWMyz * confz[flip_coo2D(jb, i-1)];
  hz = BXPzx * confx[flip_coo2D(j, ib)] + BYPzx * confx[flip_coo2D((j+1), i)] + BWPzx * confx[flip_coo2D((j+1), ib)] + BXMzx * confx[flip_coo2D(j, i-1)] + BYMzx * confx[flip_coo2D(jb, i)] + BWMzx * confx[flip_coo2D(jb, i-1)]\
     + BXPzy * confy[flip_coo2D(j, ib)] + BYPzy * confy[flip_coo2D((j+1), i)] + BWPzy * confy[flip_coo2D((j+1), ib)] + BXMzy * confy[flip_coo2D(j, i-1)] + BYMzy * confy[flip_coo2D(jb, i)] + BWMzy * confy[flip_coo2D(jb, i-1)]\
     + BXPzz * confz[flip_coo2D(j, ib)] + BYPzz * confz[flip_coo2D((j+1), i)] + BWPzz * confz[flip_coo2D((j+1), ib)] + BXMzz * confz[flip_coo2D(j, i-1)] + BYMzz * confz[flip_coo2D(jb, i)] + BWMzz * confz[flip_coo2D(jb, i-1)] + H;
  du = - confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //...
  //..0
  i = tx + 2;
  j = ty + 2;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[flip_coo2D(j, ib)] + BYPxx * confx[flip_coo2D(jb, i)] + BWPxx * confx[flip_coo2D(jb, ib)] + BXMxx * confx[flip_coo2D(j, i-1)] + BYMxx * confx[flip_coo2D(j-1, i)] + BWMxx * confx[flip_coo2D(j-1, i-1)]\
     + BXPxy * confy[flip_coo2D(j, ib)] + BYPxy * confy[flip_coo2D(jb, i)] + BWPxy * confy[flip_coo2D(jb, ib)] + BXMxy * confy[flip_coo2D(j, i-1)] + BYMxy * confy[flip_coo2D(j-1, i)] + BWMxy * confy[flip_coo2D(j-1, i-1)]\
     + BXPxz * confz[flip_coo2D(j, ib)] + BYPxz * confz[flip_coo2D(jb, i)] + BWPxz * confz[flip_coo2D(jb, ib)] + BXMxz * confz[flip_coo2D(j, i-1)] + BYMxz * confz[flip_coo2D(j-1, i)] + BWMxz * confz[flip_coo2D(j-1, i-1)];
  hy = BXPyx * confx[flip_coo2D(j, ib)] + BYPyx * confx[flip_coo2D(jb, i)] + BWPyx * confx[flip_coo2D(jb, ib)] + BXMyx * confx[flip_coo2D(j, i-1)] + BYMyx * confx[flip_coo2D(j-1, i)] + BWMyx * confx[flip_coo2D(j-1, i-1)]\
     + BXPyy * confy[flip_coo2D(j, ib)] + BYPyy * confy[flip_coo2D(jb, i)] + BWPyy * confy[flip_coo2D(jb, ib)] + BXMyy * confy[flip_coo2D(j, i-1)] + BYMyy * confy[flip_coo2D(j-1, i)] + BWMyy * confy[flip_coo2D(j-1, i-1)]\
     + BXPyz * confz[flip_coo2D(j, ib)] + BYPyz * confz[flip_coo2D(jb, i)] + BWPyz * confz[flip_coo2D(jb, ib)] + BXMyz * confz[flip_coo2D(j, i-1)] + BYMyz * confz[flip_coo2D(j-1, i)] + BWMyz * confz[flip_coo2D(j-1, i-1)];
  hz = BXPzx * confx[flip_coo2D(j, ib)] + BYPzx * confx[flip_coo2D(jb, i)] + BWPzx * confx[flip_coo2D(jb, ib)] + BXMzx * confx[flip_coo2D(j, i-1)] + BYMzx * confx[flip_coo2D(j-1, i)] + BWMzx * confx[flip_coo2D(j-1, i-1)]\
     + BXPzy * confy[flip_coo2D(j, ib)] + BYPzy * confy[flip_coo2D(jb, i)] + BWPzy * confy[flip_coo2D(jb, ib)] + BXMzy * confy[flip_coo2D(j, i-1)] + BYMzy * confy[flip_coo2D(j-1, i)] + BWMzy * confy[flip_coo2D(j-1, i-1)]\
     + BXPzz * confz[flip_coo2D(j, ib)] + BYPzz * confz[flip_coo2D(jb, i)] + BWPzz * confz[flip_coo2D(jb, ib)] + BXMzz * confz[flip_coo2D(j, i-1)] + BYMzz * confz[flip_coo2D(j-1, i)] + BWMzz * confz[flip_coo2D(j-1, i-1)] + H;
  du = -confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}


__global__ void flip3_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
  //Energy variables
  __shared__ unsigned rngShmem[1024];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float Pparameter = Pparameters[blockIdx.x / flip_BN];
  unsigned int r;
  float du;	//-dE
  float sx, sy, sz;
  float th,phi;
  float hx, hy, hz;
  //float norm;
  const int x = threadIdx.x % (flip_BlockSize_x);
  const int y = (threadIdx.x / flip_BlockSize_x);// % flip_BlockSize_y;
  const int tx = 3 * (((blockIdx.x % flip_BN) % flip_GridSize_x) * flip_BlockSize_x + x);
  const int ty = (blockIdx.x / flip_BN) * flip_SpinSize + 3 * (((blockIdx.x % flip_BN) / flip_GridSize_x) * flip_BlockSize_y + y);
  int i, j, ib, jb;
  //----------Spin flip at the bottom and left corner of each thread sqare----------
  i = tx;
  j = ty + 2;
  ib = (i + flip_SpinSize - 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  //...
  //...
  //0..
  hx = BXPxx * confx[flip_coo2D(j, i+1)] + BYPxx * confx[flip_coo2D(jb, i)] + BWPxx * confx[flip_coo2D(jb, i+1)] + BXMxx * confx[flip_coo2D(j, ib)] + BYMxx * confx[flip_coo2D((j-1), i)] + BWMxx * confx[flip_coo2D((j-1), ib)]\
     + BXPxy * confy[flip_coo2D(j, i+1)] + BYPxy * confy[flip_coo2D(jb, i)] + BWPxy * confy[flip_coo2D(jb, i+1)] + BXMxy * confy[flip_coo2D(j, ib)] + BYMxy * confy[flip_coo2D((j-1), i)] + BWMxy * confy[flip_coo2D((j-1), ib)]\
     + BXPxz * confz[flip_coo2D(j, i+1)] + BYPxz * confz[flip_coo2D(jb, i)] + BWPxz * confz[flip_coo2D(jb, i+1)] + BXMxz * confz[flip_coo2D(j, ib)] + BYMxz * confz[flip_coo2D((j-1), i)] + BWMxz * confz[flip_coo2D((j-1), ib)];
  hy = BXPyx * confx[flip_coo2D(j, i+1)] + BYPyx * confx[flip_coo2D(jb, i)] + BWPyx * confx[flip_coo2D(jb, i+1)] + BXMyx * confx[flip_coo2D(j, ib)] + BYMyx * confx[flip_coo2D((j-1), i)] + BWMyx * confx[flip_coo2D((j-1), ib)]\
     + BXPyy * confy[flip_coo2D(j, i+1)] + BYPyy * confy[flip_coo2D(jb, i)] + BWPyy * confy[flip_coo2D(jb, i+1)] + BXMyy * confy[flip_coo2D(j, ib)] + BYMyy * confy[flip_coo2D((j-1), i)] + BWMyy * confy[flip_coo2D((j-1), ib)]\
     + BXPyz * confz[flip_coo2D(j, i+1)] + BYPyz * confz[flip_coo2D(jb, i)] + BWPyz * confz[flip_coo2D(jb, i+1)] + BXMyz * confz[flip_coo2D(j, ib)] + BYMyz * confz[flip_coo2D((j-1), i)] + BWMyz * confz[flip_coo2D((j-1), ib)];
  hz = BXPzx * confx[flip_coo2D(j, i+1)] + BYPzx * confx[flip_coo2D(jb, i)] + BWPzx * confx[flip_coo2D(jb, i+1)] + BXMzx * confx[flip_coo2D(j, ib)] + BYMzx * confx[flip_coo2D((j-1), i)] + BWMzx * confx[flip_coo2D((j-1), ib)]\
     + BXPzy * confy[flip_coo2D(j, i+1)] + BYPzy * confy[flip_coo2D(jb, i)] + BWPzy * confy[flip_coo2D(jb, i+1)] + BXMzy * confy[flip_coo2D(j, ib)] + BYMzy * confy[flip_coo2D((j-1), i)] + BWMzy * confy[flip_coo2D((j-1), ib)]\
     + BXPzz * confz[flip_coo2D(j, i+1)] + BYPzz * confz[flip_coo2D(jb, i)] + BWPzz * confz[flip_coo2D(jb, i+1)] + BWMzz * confz[flip_coo2D(j, ib)] + BYMzz * confz[flip_coo2D((j-1), i)] + BWMzz * confz[flip_coo2D((j-1), ib)] + H;
  du = - confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //----------Spin flip at the top and right corner of each thread sqare----------
  i = tx + 2;
  j = ty;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == 0)	jb = j + flip_SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  //..0
  //...
  //...
  hx = BXPxx * confx[flip_coo2D(j, ib)] + BYPxx * confx[flip_coo2D((j+1), i)] + BWPxx * confx[flip_coo2D((j+1), ib)] + BXMxx * confx[flip_coo2D(j, i-1)] + BYMxx * confx[flip_coo2D(jb, i)] + BWMxx * confx[flip_coo2D(jb, i-1)]\
     + BXPxy * confy[flip_coo2D(j, ib)] + BYPxy * confy[flip_coo2D((j+1), i)] + BWPxy * confy[flip_coo2D((j+1), ib)] + BXMxy * confy[flip_coo2D(j, i-1)] + BYMxy * confy[flip_coo2D(jb, i)] + BWMxy * confy[flip_coo2D(jb, i-1)]\
     + BXPxz * confz[flip_coo2D(j, ib)] + BYPxz * confz[flip_coo2D((j+1), i)] + BWPxz * confz[flip_coo2D((j+1), ib)] + BXMxz * confz[flip_coo2D(j, i-1)] + BYMxz * confz[flip_coo2D(jb, i)] + BWMxz * confz[flip_coo2D(jb, i-1)];
  hy = BXPyx * confx[flip_coo2D(j, ib)] + BYPyx * confx[flip_coo2D((j+1), i)] + BWPyx * confx[flip_coo2D((j+1), ib)] + BXMyx * confx[flip_coo2D(j, i-1)] + BYMyx * confx[flip_coo2D(jb, i)] + BWMyx * confx[flip_coo2D(jb, i-1)]\
     + BXPyy * confy[flip_coo2D(j, ib)] + BYPyy * confy[flip_coo2D((j+1), i)] + BWPyy * confy[flip_coo2D((j+1), ib)] + BXMyy * confy[flip_coo2D(j, i-1)] + BYMyy * confy[flip_coo2D(jb, i)] + BWMyy * confy[flip_coo2D(jb, i-1)]\
     + BXPyz * confz[flip_coo2D(j, ib)] + BYPyz * confz[flip_coo2D((j+1), i)] + BWPyz * confz[flip_coo2D((j+1), ib)] + BXMyz * confz[flip_coo2D(j, i-1)] + BYMyz * confz[flip_coo2D(jb, i)] + BWMyz * confz[flip_coo2D(jb, i-1)];
  hz = BXPzx * confx[flip_coo2D(j, ib)] + BYPzx * confx[flip_coo2D((j+1), i)] + BWPzx * confx[flip_coo2D((j+1), ib)] + BXMzx * confx[flip_coo2D(j, i-1)] + BYMzx * confx[flip_coo2D(jb, i)] + BWMzx * confx[flip_coo2D(jb, i-1)]\
     + BXPzy * confy[flip_coo2D(j, ib)] + BYPzy * confy[flip_coo2D((j+1), i)] + BWPzy * confy[flip_coo2D((j+1), ib)] + BXMzy * confy[flip_coo2D(j, i-1)] + BYMzy * confy[flip_coo2D(jb, i)] + BWMzy * confy[flip_coo2D(jb, i-1)]\
     + BXPzz * confz[flip_coo2D(j, ib)] + BYPzz * confz[flip_coo2D((j+1), i)] + BWPzz * confz[flip_coo2D((j+1), ib)] + BXMzz * confz[flip_coo2D(j, i-1)] + BYMzz * confz[flip_coo2D(jb, i)] + BWMzz * confz[flip_coo2D(jb, i-1)] + H;
  du = - confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //.0.
  //...
  i = tx + 1;
  j = ty + 1;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[flip_coo2D(j, ib)] + BYPxx * confx[flip_coo2D(jb, i)] + BWPxx * confx[flip_coo2D(jb, ib)] + BXMxx * confx[flip_coo2D(j, i-1)] + BYMxx * confx[flip_coo2D(j-1, i)] + BWMxx * confx[flip_coo2D(j-1, i-1)]\
     + BXPxy * confy[flip_coo2D(j, ib)] + BYPxy * confy[flip_coo2D(jb, i)] + BWPxy * confy[flip_coo2D(jb, ib)] + BXMxy * confy[flip_coo2D(j, i-1)] + BYMxy * confy[flip_coo2D(j-1, i)] + BWMxy * confy[flip_coo2D(j-1, i-1)]\
     + BXPxz * confz[flip_coo2D(j, ib)] + BYPxz * confz[flip_coo2D(jb, i)] + BWPxz * confz[flip_coo2D(jb, ib)] + BXMxz * confz[flip_coo2D(j, i-1)] + BYMxz * confz[flip_coo2D(j-1, i)] + BWMxz * confz[flip_coo2D(j-1, i-1)];
  hy = BXPyx * confx[flip_coo2D(j, ib)] + BYPyx * confx[flip_coo2D(jb, i)] + BWPyx * confx[flip_coo2D(jb, ib)] + BXMyx * confx[flip_coo2D(j, i-1)] + BYMyx * confx[flip_coo2D(j-1, i)] + BWMyx * confx[flip_coo2D(j-1, i-1)]\
     + BXPyy * confy[flip_coo2D(j, ib)] + BYPyy * confy[flip_coo2D(jb, i)] + BWPyy * confy[flip_coo2D(jb, ib)] + BXMyy * confy[flip_coo2D(j, i-1)] + BYMyy * confy[flip_coo2D(j-1, i)] + BWMyy * confy[flip_coo2D(j-1, i-1)]\
     + BXPyz * confz[flip_coo2D(j, ib)] + BYPyz * confz[flip_coo2D(jb, i)] + BWPyz * confz[flip_coo2D(jb, ib)] + BXMyz * confz[flip_coo2D(j, i-1)] + BYMyz * confz[flip_coo2D(j-1, i)] + BWMyz * confz[flip_coo2D(j-1, i-1)];
  hz = BXPzx * confx[flip_coo2D(j, ib)] + BYPzx * confx[flip_coo2D(jb, i)] + BWPzx * confx[flip_coo2D(jb, ib)] + BXMzx * confx[flip_coo2D(j, i-1)] + BYMzx * confx[flip_coo2D(j-1, i)] + BWMzx * confx[flip_coo2D(j-1, i-1)]\
     + BXPzy * confy[flip_coo2D(j, ib)] + BYPzy * confy[flip_coo2D(jb, i)] + BWPzy * confy[flip_coo2D(jb, ib)] + BXMzy * confy[flip_coo2D(j, i-1)] + BYMzy * confy[flip_coo2D(j-1, i)] + BWMzy * confy[flip_coo2D(j-1, i-1)]\
     + BXPzz * confz[flip_coo2D(j, ib)] + BYPzz * confz[flip_coo2D(jb, i)] + BWPzz * confz[flip_coo2D(jb, ib)] + BXMzz * confz[flip_coo2D(j, i-1)] + BYMzz * confz[flip_coo2D(j-1, i)] + BWMzz * confz[flip_coo2D(j-1, i-1)] + H;
  du = -confx[flip_coo2D(j, i)] * hx - confy[flip_coo2D(j, i)] * hy - confz[flip_coo2D(j, i)] * hz + flip_A * confz[flip_coo2D(j, i)] * confz[flip_coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - flip_A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo2D(j, i)] = sx;
    confy[flip_coo2D(j, i)] = sy;
    confz[flip_coo2D(j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}
#endif
