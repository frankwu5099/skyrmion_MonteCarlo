#ifdef SQ
#include "updates.cuh"
__global__ void flipTLBR_2D(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
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
  //----------Spin flip at the top and left corner of each thread sqare----------
  i = tx;
  j = ty;
  ib = (i + SpinSize - 1) % SpinSize;
  if((j % SpinSize) == 0)	jb = j + SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, i+1)] + BYPxx * confx[coo2D((j+1), i)] + BXMxx * confx[coo2D(j, ib)] + BYMxx * confx[coo2D(jb, i)]\
     + BXPxy * confy[coo2D(j, i+1)] + BYPxy * confy[coo2D((j+1), i)] + BXMxy * confy[coo2D(j, ib)] + BYMxy * confy[coo2D(jb, i)]\
     + BXPxz * confz[coo2D(j, i+1)] + BYPxz * confz[coo2D((j+1), i)] + BXMxz * confz[coo2D(j, ib)] + BYMxz * confz[coo2D(jb, i)];
  hy = BXPyx * confx[coo2D(j, i+1)] + BYPyx * confx[coo2D((j+1), i)] + BXMyx * confx[coo2D(j, ib)] + BYMyx * confx[coo2D(jb, i)]\
     + BXPyy * confy[coo2D(j, i+1)] + BYPyy * confy[coo2D((j+1), i)] + BXMyy * confy[coo2D(j, ib)] + BYMyy * confy[coo2D(jb, i)]\
     + BXPyz * confz[coo2D(j, i+1)] + BYPyz * confz[coo2D((j+1), i)] + BXMyz * confz[coo2D(j, ib)] + BYMyz * confz[coo2D(jb, i)];
  hz = BXPzx * confx[coo2D(j, i+1)] + BYPzx * confx[coo2D((j+1), i)] + BXMzx * confx[coo2D(j, ib)] + BYMzx * confx[coo2D(jb, i)]\
     + BXPzy * confy[coo2D(j, i+1)] + BYPzy * confy[coo2D((j+1), i)] + BXMzy * confy[coo2D(j, ib)] + BYMzy * confy[coo2D(jb, i)]\
     + BXPzz * confz[coo2D(j, i+1)] + BYPzz * confz[coo2D((j+1), i)] + BXMzz * confz[coo2D(j, ib)] + BYMzz * confz[coo2D(jb, i)] + H;
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

  //----------Spin flip at the bottom and right corner of each thread sqare----------
  i = tx + 1;
  j = ty + 1;
  ib = (i + 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D(jb, i)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D((j-1), i)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D(jb, i)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D((j-1), i)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D(jb, i)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D((j-1), i)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D(jb, i)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D((j-1), i)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D(jb, i)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D((j-1), i)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D(jb, i)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D((j-1), i)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D(jb, i)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D((j-1), i)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D(jb, i)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D((j-1), i)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D(jb, i)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D((j-1), i)] + H;
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

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}



__global__ void flipBLTR_2D(float *confx, float *confy, float *confz, unsigned int *rngState, float* Pparameters, float Cparameter){
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
  const int tx = 2 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
  const int ty = (blockIdx.x / BN) * SpinSize + 2 * (((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
  int i, j, ib, jb;
  //----------Spin flip at the bottom and left corner of each thread sqare----------
  i = tx;
  j = ty + 1;
  ib = (i + SpinSize - 1) % SpinSize;
  if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  hx = BXPxx * confx[coo2D(j, i+1)] + BYPxx * confx[coo2D(jb, i)] + BXMxx * confx[coo2D(j, ib)] + BYMxx * confx[coo2D((j-1), i)]\
     + BXPxy * confy[coo2D(j, i+1)] + BYPxy * confy[coo2D(jb, i)] + BXMxy * confy[coo2D(j, ib)] + BYMxy * confy[coo2D((j-1), i)]\
     + BXPxz * confz[coo2D(j, i+1)] + BYPxz * confz[coo2D(jb, i)] + BXMxz * confz[coo2D(j, ib)] + BYMxz * confz[coo2D((j-1), i)];
  hy = BXPyx * confx[coo2D(j, i+1)] + BYPyx * confx[coo2D(jb, i)] + BXMyx * confx[coo2D(j, ib)] + BYMyx * confx[coo2D((j-1), i)]\
     + BXPyy * confy[coo2D(j, i+1)] + BYPyy * confy[coo2D(jb, i)] + BXMyy * confy[coo2D(j, ib)] + BYMyy * confy[coo2D((j-1), i)]\
     + BXPyz * confz[coo2D(j, i+1)] + BYPyz * confz[coo2D(jb, i)] + BXMyz * confz[coo2D(j, ib)] + BYMyz * confz[coo2D((j-1), i)];
  hz = BXPzx * confx[coo2D(j, i+1)] + BYPzx * confx[coo2D(jb, i)] + BXMzx * confx[coo2D(j, ib)] + BYMzx * confx[coo2D((j-1), i)]\
     + BXPzy * confy[coo2D(j, i+1)] + BYPzy * confy[coo2D(jb, i)] + BXMzy * confy[coo2D(j, ib)] + BYMzy * confy[coo2D((j-1), i)]\
     + BXPzz * confz[coo2D(j, i+1)] + BYPzz * confz[coo2D(jb, i)] + BXMzz * confz[coo2D(j, ib)] + BYMzz * confz[coo2D((j-1), i)] + H;
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
  hx = BXPxx * confx[coo2D(j, ib)] + BYPxx * confx[coo2D((j+1), i)] + BXMxx * confx[coo2D(j, i-1)] + BYMxx * confx[coo2D(jb, i)]\
     + BXPxy * confy[coo2D(j, ib)] + BYPxy * confy[coo2D((j+1), i)] + BXMxy * confy[coo2D(j, i-1)] + BYMxy * confy[coo2D(jb, i)]\
     + BXPxz * confz[coo2D(j, ib)] + BYPxz * confz[coo2D((j+1), i)] + BXMxz * confz[coo2D(j, i-1)] + BYMxz * confz[coo2D(jb, i)];
  hy = BXPyx * confx[coo2D(j, ib)] + BYPyx * confx[coo2D((j+1), i)] + BXMyx * confx[coo2D(j, i-1)] + BYMyx * confx[coo2D(jb, i)]\
     + BXPyy * confy[coo2D(j, ib)] + BYPyy * confy[coo2D((j+1), i)] + BXMyy * confy[coo2D(j, i-1)] + BYMyy * confy[coo2D(jb, i)]\
     + BXPyz * confz[coo2D(j, ib)] + BYPyz * confz[coo2D((j+1), i)] + BXMyz * confz[coo2D(j, i-1)] + BYMyz * confz[coo2D(jb, i)];
  hz = BXPzx * confx[coo2D(j, ib)] + BYPzx * confx[coo2D((j+1), i)] + BXMzx * confx[coo2D(j, i-1)] + BYMzx * confx[coo2D(jb, i)]\
     + BXPzy * confy[coo2D(j, ib)] + BYPzy * confy[coo2D((j+1), i)] + BXMzy * confy[coo2D(j, i-1)] + BYMzy * confy[coo2D(jb, i)]\
     + BXPzz * confz[coo2D(j, ib)] + BYPzz * confz[coo2D((j+1), i)] + BXMzz * confz[coo2D(j, i-1)] + BYMzz * confz[coo2D(jb, i)] + H;
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

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}
#endif
