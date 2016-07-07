#include "updates.cuh"
#ifdef TRI
__global__ void flip1_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
  //Energy variables
  extern __shared__ unsigned rngShmem[];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float Pparameter = Pparameters[blockIdx.x / BN];
  unsigned int r;
  float du;	//-dE
  float sx, sy, sz;
  float th,phi;
  float hx, hy, hz;
  //float norm;
  const int x = threadIdx.x % (BlockSize_x);
  const int y = (threadIdx.x / BlockSize_x);
  const int tx = 3 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
  const int ty =(blockIdx.x / BN) * SpinSize +  3 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
  int i, j, ib, jb;
  //0..
  //...
  //...
  i = tx;
  j = ty;
  ib = (i + SpinSize - 1) % SpinSize;
  if((j % SpinSize) == 0)	jb = j + SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, i+1)] + BYPxx * confx[coo2D(j+1, i)] + BWPxx * confx[coo2D(j+1, i+1)] + BXMxx * confx[coo2D(j, ib)] + BYMxx * confx[coo2D(jb, i)] + BWMxx * confx[coo2D(jb, ib)]\
     + BXPxy * confy[coo2D(j, i+1)] + BYPxy * confy[coo2D(j+1, i)] + BWPxy * confy[coo2D(j+1, i+1)] + BXMxy * confy[coo2D(j, ib)] + BYMxy * confy[coo2D(jb, i)] + BWMxy * confy[coo2D(jb, ib)]\
     + BXPxz * confz[coo2D(j, i+1)] + BYPxz * confz[coo2D(j+1, i)] + BWPxz * confz[coo2D(j+1, i+1)] + BXMxz * confz[coo2D(j, ib)] + BYMxz * confz[coo2D(jb, i)] + BWMxz * confz[coo2D(jb, ib)];
  hy = BXPyx * confx[coo2D(j, i+1)] + BYPyx * confx[coo2D(j+1, i)] + BWPyx * confx[coo2D(j+1, i+1)] + BXMyx * confx[coo2D(j, ib)] + BYMyx * confx[coo2D(jb, i)] + BWMyx * confx[coo2D(jb, ib)]\
     + BXPyy * confy[coo2D(j, i+1)] + BYPyy * confy[coo2D(j+1, i)] + BWPyy * confy[coo2D(j+1, i+1)] + BXMyy * confy[coo2D(j, ib)] + BYMyy * confy[coo2D(jb, i)] + BWMyy * confy[coo2D(jb, ib)]\
     + BXPyz * confz[coo2D(j, i+1)] + BYPyz * confz[coo2D(j+1, i)] + BWPyz * confz[coo2D(j+1, i+1)] + BXMyz * confz[coo2D(j, ib)] + BYMyz * confz[coo2D(jb, i)] + BWMyz * confz[coo2D(jb, ib)];
  hz = BXPzx * confx[coo2D(j, i+1)] + BYPzx * confx[coo2D(j+1, i)] + BWPzx * confx[coo2D(j+1, i+1)] + BXMzx * confx[coo2D(j, ib)] + BYMzx * confx[coo2D(jb, i)] + BWMzx * confx[coo2D(jb, ib)]\
     + BXPzy * confy[coo2D(j, i+1)] + BYPzy * confy[coo2D(j+1, i)] + BWPzy * confy[coo2D(j+1, i+1)] + BXMzy * confy[coo2D(j, ib)] + BYMzy * confy[coo2D(jb, i)] + BWMzy * confy[coo2D(jb, ib)]\
     + BXPzz * confz[coo2D(j, i+1)] + BYPzz * confz[coo2D(j+1, i)] + BWPzz * confz[coo2D(j+1, i+1)] + BXMzz * confz[coo2D(j, ib)] + BYMzz * confz[coo2D(jb, i)] + BWMzz * confz[coo2D(jb, ib)] + H;
  du = - confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //..0
  //...
  i = tx + 2;
  j = ty + 1;
  ib = (i + 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D(jb, i)] + BWPxx * confx[coo2D(jb, ib)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D(j-1, i)] + BWMxx * confx[coo2D(j-1, i-1)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D(jb, i)] + BWPxy * confy[coo2D(jb, ib)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D(j-1, i)] + BWMxy * confy[coo2D(j-1, i-1)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D(jb, i)] + BWPxz * confz[coo2D(jb, ib)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D(j-1, i)] + BWMxz * confz[coo2D(j-1, i-1)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D(jb, i)] + BWPyx * confx[coo2D(jb, ib)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D(j-1, i)] + BWMyx * confx[coo2D(j-1, i-1)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D(jb, i)] + BWPyy * confy[coo2D(jb, ib)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D(j-1, i)] + BWMyy * confy[coo2D(j-1, i-1)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D(jb, i)] + BWPyz * confz[coo2D(jb, ib)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D(j-1, i)] + BWMyz * confz[coo2D(j-1, i-1)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D(jb, i)] + BWPzx * confx[coo2D(jb, ib)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D(j-1, i)] + BWMzx * confx[coo2D(j-1, i-1)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D(jb, i)] + BWPzy * confy[coo2D(jb, ib)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D(j-1, i)] + BWMzy * confy[coo2D(j-1, i-1)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D(jb, i)] + BWPzz * confz[coo2D(jb, ib)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D(j-1, i)] + BWMzz * confz[coo2D(j-1, i-1)] + H;
  du = -confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //...
  //.0.
  i = tx + 1;
  j = ty + 2;
  ib = (i + 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D(jb, i)] + BWPxx * confx[coo2D(jb, ib)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D(j-1, i)] + BWMxx * confx[coo2D(j-1, i-1)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D(jb, i)] + BWPxy * confy[coo2D(jb, ib)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D(j-1, i)] + BWMxy * confy[coo2D(j-1, i-1)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D(jb, i)] + BWPxz * confz[coo2D(jb, ib)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D(j-1, i)] + BWMxz * confz[coo2D(j-1, i-1)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D(jb, i)] + BWPyx * confx[coo2D(jb, ib)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D(j-1, i)] + BWMyx * confx[coo2D(j-1, i-1)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D(jb, i)] + BWPyy * confy[coo2D(jb, ib)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D(j-1, i)] + BWMyy * confy[coo2D(j-1, i-1)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D(jb, i)] + BWPyz * confz[coo2D(jb, ib)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D(j-1, i)] + BWMyz * confz[coo2D(j-1, i-1)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D(jb, i)] + BWPzx * confx[coo2D(jb, ib)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D(j-1, i)] + BWMzx * confx[coo2D(j-1, i-1)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D(jb, i)] + BWPzy * confy[coo2D(jb, ib)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D(j-1, i)] + BWMzy * confy[coo2D(j-1, i-1)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D(jb, i)] + BWPzz * confz[coo2D(jb, ib)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D(j-1, i)] + BWMzz * confz[coo2D(j-1, i-1)] + H;
  du = -confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}



__global__ void flip2_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
  //Energy variables
  extern __shared__ unsigned rngShmem[];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float Pparameter = Pparameters[blockIdx.x / BN];
  unsigned int r;
  float du;	//-dE
  float sx, sy, sz;
  float th,phi;
  float hx, hy, hz;
  //float norm;
  const int x = threadIdx.x % (BlockSize_x);
  const int y = (threadIdx.x / BlockSize_x);// % BlockSize_y;
  const int tx = 3 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
  const int ty = (blockIdx.x / BN) * SpinSize + 3 * (((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
  int i, j, ib, jb;
  //----------Spin flip at the bottom and left corner of each thread sqare----------
  i = tx;
  j = ty + 1;
  ib = (i + SpinSize - 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  //...
  //0..
  //...
  hx = BXPxx * confx[coo2D(j, i+1)] + BYPxx * confx[coo2D(jb, i)] + BWPxx * confx[coo2D(jb, i+1)] + BXMxx * confx[coo2D(j, ib)] + BYMxx * confx[coo2D((j-1), i)] + BWMxx * confx[coo2D((j-1), ib)]\
     + BXPxy * confy[coo2D(j, i+1)] + BYPxy * confy[coo2D(jb, i)] + BWPxy * confy[coo2D(jb, i+1)] + BXMxy * confy[coo2D(j, ib)] + BYMxy * confy[coo2D((j-1), i)] + BWMxy * confy[coo2D((j-1), ib)]\
     + BXPxz * confz[coo2D(j, i+1)] + BYPxz * confz[coo2D(jb, i)] + BWPxz * confz[coo2D(jb, i+1)] + BXMxz * confz[coo2D(j, ib)] + BYMxz * confz[coo2D((j-1), i)] + BWMxz * confz[coo2D((j-1), ib)];
  hy = BXPyx * confx[coo2D(j, i+1)] + BYPyx * confx[coo2D(jb, i)] + BWPyx * confx[coo2D(jb, i+1)] + BXMyx * confx[coo2D(j, ib)] + BYMyx * confx[coo2D((j-1), i)] + BWMyx * confx[coo2D((j-1), ib)]\
     + BXPyy * confy[coo2D(j, i+1)] + BYPyy * confy[coo2D(jb, i)] + BWPyy * confy[coo2D(jb, i+1)] + BXMyy * confy[coo2D(j, ib)] + BYMyy * confy[coo2D((j-1), i)] + BWMyy * confy[coo2D((j-1), ib)]\
     + BXPyz * confz[coo2D(j, i+1)] + BYPyz * confz[coo2D(jb, i)] + BWPyz * confz[coo2D(jb, i+1)] + BXMyz * confz[coo2D(j, ib)] + BYMyz * confz[coo2D((j-1), i)] + BWMyz * confz[coo2D((j-1), ib)];
  hz = BXPzx * confx[coo2D(j, i+1)] + BYPzx * confx[coo2D(jb, i)] + BWPzx * confx[coo2D(jb, i+1)] + BXMzx * confx[coo2D(j, ib)] + BYMzx * confx[coo2D((j-1), i)] + BWMzx * confx[coo2D((j-1), ib)]\
     + BXPzy * confy[coo2D(j, i+1)] + BYPzy * confy[coo2D(jb, i)] + BWPzy * confy[coo2D(jb, i+1)] + BXMzy * confy[coo2D(j, ib)] + BYMzy * confy[coo2D((j-1), i)] + BWMzy * confy[coo2D((j-1), ib)]\
     + BXPzz * confz[coo2D(j, i+1)] + BYPzz * confz[coo2D(jb, i)] + BWPzz * confz[coo2D(jb, i+1)] + BWMzz * confz[coo2D(j, ib)] + BYMzz * confz[coo2D((j-1), i)] + BWMzz * confz[coo2D((j-1), ib)] + H;
  du = - confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //----------Spin flip at the top and right corner of each thread sqare----------
  i = tx + 1;
  j = ty;
  ib = (i + 1) % SpinSize;
  if((j % SpinSize) == 0)	jb = j + SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  //.0.
  //...
  //...
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D((j+1), i)] + BWPxx * confx[coo2D((j+1), ib)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D(jb, i)] + BWMxx * confx[coo2D(jb, i-1)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D((j+1), i)] + BWPxy * confy[coo2D((j+1), ib)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D(jb, i)] + BWMxy * confy[coo2D(jb, i-1)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D((j+1), i)] + BWPxz * confz[coo2D((j+1), ib)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D(jb, i)] + BWMxz * confz[coo2D(jb, i-1)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D((j+1), i)] + BWPyx * confx[coo2D((j+1), ib)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D(jb, i)] + BWMyx * confx[coo2D(jb, i-1)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D((j+1), i)] + BWPyy * confy[coo2D((j+1), ib)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D(jb, i)] + BWMyy * confy[coo2D(jb, i-1)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D((j+1), i)] + BWPyz * confz[coo2D((j+1), ib)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D(jb, i)] + BWMyz * confz[coo2D(jb, i-1)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D((j+1), i)] + BWPzx * confx[coo2D((j+1), ib)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D(jb, i)] + BWMzx * confx[coo2D(jb, i-1)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D((j+1), i)] + BWPzy * confy[coo2D((j+1), ib)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D(jb, i)] + BWMzy * confy[coo2D(jb, i-1)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D((j+1), i)] + BWPzz * confz[coo2D((j+1), ib)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D(jb, i)] + BWMzz * confz[coo2D(jb, i-1)] + H;
  du = - confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //...
  //..0
  i = tx + 2;
  j = ty + 2;
  ib = (i + 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D(jb, i)] + BWPxx * confx[coo2D(jb, ib)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D(j-1, i)] + BWMxx * confx[coo2D(j-1, i-1)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D(jb, i)] + BWPxy * confy[coo2D(jb, ib)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D(j-1, i)] + BWMxy * confy[coo2D(j-1, i-1)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D(jb, i)] + BWPxz * confz[coo2D(jb, ib)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D(j-1, i)] + BWMxz * confz[coo2D(j-1, i-1)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D(jb, i)] + BWPyx * confx[coo2D(jb, ib)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D(j-1, i)] + BWMyx * confx[coo2D(j-1, i-1)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D(jb, i)] + BWPyy * confy[coo2D(jb, ib)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D(j-1, i)] + BWMyy * confy[coo2D(j-1, i-1)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D(jb, i)] + BWPyz * confz[coo2D(jb, ib)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D(j-1, i)] + BWMyz * confz[coo2D(j-1, i-1)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D(jb, i)] + BWPzx * confx[coo2D(jb, ib)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D(j-1, i)] + BWMzx * confx[coo2D(j-1, i-1)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D(jb, i)] + BWPzy * confy[coo2D(jb, ib)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D(j-1, i)] + BWMzy * confy[coo2D(j-1, i-1)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D(jb, i)] + BWPzz * confz[coo2D(jb, ib)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D(j-1, i)] + BWMzz * confz[coo2D(j-1, i-1)] + H;
  du = -confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}


__global__ void flip3_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
  //Energy variables
  extern __shared__ unsigned rngShmem[];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float Pparameter = Pparameters[blockIdx.x / BN];
  unsigned int r;
  float du;	//-dE
  float sx, sy, sz;
  float th,phi;
  float hx, hy, hz;
  //float norm;
  const int x = threadIdx.x % (BlockSize_x);
  const int y = (threadIdx.x / BlockSize_x);// % BlockSize_y;
  const int tx = 3 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
  const int ty = (blockIdx.x / BN) * SpinSize + 3 * (((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
  int i, j, ib, jb;
  //----------Spin flip at the bottom and left corner of each thread sqare----------
  i = tx;
  j = ty + 2;
  ib = (i + SpinSize - 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  //...
  //...
  //0..
  hx = BXPxx * confx[coo2D(j, i+1)] + BYPxx * confx[coo2D(jb, i)] + BWPxx * confx[coo2D(jb, i+1)] + BXMxx * confx[coo2D(j, ib)] + BYMxx * confx[coo2D((j-1), i)] + BWMxx * confx[coo2D((j-1), ib)]\
     + BXPxy * confy[coo2D(j, i+1)] + BYPxy * confy[coo2D(jb, i)] + BWPxy * confy[coo2D(jb, i+1)] + BXMxy * confy[coo2D(j, ib)] + BYMxy * confy[coo2D((j-1), i)] + BWMxy * confy[coo2D((j-1), ib)]\
     + BXPxz * confz[coo2D(j, i+1)] + BYPxz * confz[coo2D(jb, i)] + BWPxz * confz[coo2D(jb, i+1)] + BXMxz * confz[coo2D(j, ib)] + BYMxz * confz[coo2D((j-1), i)] + BWMxz * confz[coo2D((j-1), ib)];
  hy = BXPyx * confx[coo2D(j, i+1)] + BYPyx * confx[coo2D(jb, i)] + BWPyx * confx[coo2D(jb, i+1)] + BXMyx * confx[coo2D(j, ib)] + BYMyx * confx[coo2D((j-1), i)] + BWMyx * confx[coo2D((j-1), ib)]\
     + BXPyy * confy[coo2D(j, i+1)] + BYPyy * confy[coo2D(jb, i)] + BWPyy * confy[coo2D(jb, i+1)] + BXMyy * confy[coo2D(j, ib)] + BYMyy * confy[coo2D((j-1), i)] + BWMyy * confy[coo2D((j-1), ib)]\
     + BXPyz * confz[coo2D(j, i+1)] + BYPyz * confz[coo2D(jb, i)] + BWPyz * confz[coo2D(jb, i+1)] + BXMyz * confz[coo2D(j, ib)] + BYMyz * confz[coo2D((j-1), i)] + BWMyz * confz[coo2D((j-1), ib)];
  hz = BXPzx * confx[coo2D(j, i+1)] + BYPzx * confx[coo2D(jb, i)] + BWPzx * confx[coo2D(jb, i+1)] + BXMzx * confx[coo2D(j, ib)] + BYMzx * confx[coo2D((j-1), i)] + BWMzx * confx[coo2D((j-1), ib)]\
     + BXPzy * confy[coo2D(j, i+1)] + BYPzy * confy[coo2D(jb, i)] + BWPzy * confy[coo2D(jb, i+1)] + BXMzy * confy[coo2D(j, ib)] + BYMzy * confy[coo2D((j-1), i)] + BWMzy * confy[coo2D((j-1), ib)]\
     + BXPzz * confz[coo2D(j, i+1)] + BYPzz * confz[coo2D(jb, i)] + BWPzz * confz[coo2D(jb, i+1)] + BWMzz * confz[coo2D(j, ib)] + BYMzz * confz[coo2D((j-1), i)] + BWMzz * confz[coo2D((j-1), ib)] + H;
  du = - confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //----------Spin flip at the top and right corner of each thread sqare----------
  i = tx + 2;
  j = ty;
  ib = (i + 1) % SpinSize;
  if((j % SpinSize) == 0)	jb = j + SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  //..0
  //...
  //...
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D((j+1), i)] + BWPxx * confx[coo2D((j+1), ib)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D(jb, i)] + BWMxx * confx[coo2D(jb, i-1)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D((j+1), i)] + BWPxy * confy[coo2D((j+1), ib)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D(jb, i)] + BWMxy * confy[coo2D(jb, i-1)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D((j+1), i)] + BWPxz * confz[coo2D((j+1), ib)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D(jb, i)] + BWMxz * confz[coo2D(jb, i-1)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D((j+1), i)] + BWPyx * confx[coo2D((j+1), ib)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D(jb, i)] + BWMyx * confx[coo2D(jb, i-1)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D((j+1), i)] + BWPyy * confy[coo2D((j+1), ib)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D(jb, i)] + BWMyy * confy[coo2D(jb, i-1)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D((j+1), i)] + BWPyz * confz[coo2D((j+1), ib)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D(jb, i)] + BWMyz * confz[coo2D(jb, i-1)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D((j+1), i)] + BWPzx * confx[coo2D((j+1), ib)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D(jb, i)] + BWMzx * confx[coo2D(jb, i-1)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D((j+1), i)] + BWPzy * confy[coo2D((j+1), ib)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D(jb, i)] + BWMzy * confy[coo2D(jb, i-1)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D((j+1), i)] + BWPzz * confz[coo2D((j+1), ib)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D(jb, i)] + BWMzz * confz[coo2D(jb, i-1)] + H;
  du = - confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //...
  //.0.
  //...
  i = tx + 1;
  j = ty + 1;
  ib = (i + 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D(jb, i)] + BWPxx * confx[coo2D(jb, ib)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D(j-1, i)] + BWMxx * confx[coo2D(j-1, i-1)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D(jb, i)] + BWPxy * confy[coo2D(jb, ib)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D(j-1, i)] + BWMxy * confy[coo2D(j-1, i-1)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D(jb, i)] + BWPxz * confz[coo2D(jb, ib)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D(j-1, i)] + BWMxz * confz[coo2D(j-1, i-1)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D(jb, i)] + BWPyx * confx[coo2D(jb, ib)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D(j-1, i)] + BWMyx * confx[coo2D(j-1, i-1)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D(jb, i)] + BWPyy * confy[coo2D(jb, ib)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D(j-1, i)] + BWMyy * confy[coo2D(j-1, i-1)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D(jb, i)] + BWPyz * confz[coo2D(jb, ib)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D(j-1, i)] + BWMyz * confz[coo2D(j-1, i-1)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D(jb, i)] + BWPzx * confx[coo2D(jb, ib)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D(j-1, i)] + BWMzx * confx[coo2D(j-1, i-1)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D(jb, i)] + BWPzy * confy[coo2D(jb, ib)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D(j-1, i)] + BWMzy * confy[coo2D(j-1, i-1)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D(jb, i)] + BWPzz * confz[coo2D(jb, ib)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D(j-1, i)] + BWMzz * confz[coo2D(j-1, i-1)] + H;
  du = -confx[coo2D(j, i)] * hx - confy[coo2D(j, i)] * hy - confz[coo2D(j, i)] * hz + A * confz[coo2D(j, i)] * confz[coo2D(j, i)];
  r = WarpStandard_Generate(rngRegs, rngShmem);
  sz = r * NORM - 1;
  th = asin( sz );
  r = WarpStandard_Generate(rngRegs, rngShmem);
  phi = r*TOPI;
  sz = sin( th );
  sx = cos( th ) * cos( phi );
  sy = cos( th ) * sin( phi );
  du += sx * hx + sy * hy + sz * hz - A * sz * sz;
  r = WarpStandard_Generate(rngRegs, rngShmem);
  if(du >= 0){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo2D(j, i)] = sx;
    confy[coo2D(j, i)] = sy;
    confz[coo2D(j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}
#endif
