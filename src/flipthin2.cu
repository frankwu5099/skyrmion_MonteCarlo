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
__constant__ float BXPyz;
__constant__ float BYPyz;
__constant__ float BWPyz;
__constant__ float BXMyz;
__constant__ float BYMyz;
__constant__ float BWMyz;
__constant__ float BXPzy;
__constant__ float BYPzy;
__constant__ float BWPzy;
__constant__ float BXMzy;
__constant__ float BYMzy;
__constant__ float BWMzy;
__constant__ float BXPxz;
__constant__ float BYPxz;
__constant__ float BWPxz;
__constant__ float BXMxz;
__constant__ float BYMxz;
__constant__ float BWMxz;
__constant__ float BXPzx;
__constant__ float BYPzx;
__constant__ float BWPzx;
__constant__ float BXMzx;
__constant__ float BYMzx;
__constant__ float BWMzx;
__constant__ float BZMxy;
__constant__ float BZMyx;
__constant__ float BZPxy;
__constant__ float BZPyx;
void move_params_device_flip(){
  float tmpp;
  cudaMemcpyToSymbol( flip_SpinSize, &H_SpinSize, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_SpinSize_z, &H_SpinSize_z, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_BlockSize_x, &H_BlockSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_BlockSize_y, &H_BlockSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_GridSize_x, &H_GridSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_GridSize_y, &H_GridSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_BN, &H_BN, sizeof(unsigned int));
  cudaMemcpyToSymbol( flip_A , &H_A , sizeof(float));
  tmpp = (DD);
  cudaMemcpyToSymbol( BXPyz, &tmpp, sizeof(float));
  tmpp = (-0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( BYPyz, &tmpp, sizeof(float));
  tmpp = (0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( BWPyz, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( BXMyz, &tmpp, sizeof(float));
  tmpp = (0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( BYMyz, &tmpp, sizeof(float));
  tmpp = (-0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( BWMyz, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( BXPzy, &tmpp, sizeof(float));
  tmpp =  (0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( BYPzy, &tmpp, sizeof(float));
  tmpp =  (-0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( BWPzy, &tmpp, sizeof(float));
  tmpp = (DD);
  cudaMemcpyToSymbol( BXMzy, &tmpp, sizeof(float));
  tmpp = (-0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( BYMzy, &tmpp, sizeof(float));
  tmpp = (0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( BWMzy, &tmpp, sizeof(float));
  tmpp = (DR);
  cudaMemcpyToSymbol( BXPxz, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( BYPxz, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( BWPxz, &tmpp, sizeof(float));
  tmpp = (-DR);
  cudaMemcpyToSymbol( BXMxz, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( BYMxz, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( BWMxz, &tmpp, sizeof(float));
  tmpp = (-DR);
  cudaMemcpyToSymbol( BXPzx, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( BYPzx, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( BWPzx, &tmpp, sizeof(float));
  tmpp = (DR);
  cudaMemcpyToSymbol( BXMzx, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( BYMzx, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( BWMzx, &tmpp, sizeof(float));
  tmpp = (DD);
  cudaMemcpyToSymbol( BZPxy, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( BZPyx, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( BZMxy, &tmpp, sizeof(float));
  tmpp = (DD);
  cudaMemcpyToSymbol( BZMyx, &tmpp, sizeof(float));
}
__global__ void flip1_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float* invTs){
  //Energy variables
  extern __shared__ unsigned rngShmem[];
  //__shared__ unsigned rngShmem[1024];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float H = Hs[blockIdx.x / flip_BN];
  float invT = invTs[blockIdx.x / flip_BN];
  unsigned int r;
  float du;	//-dE
  float sx, sy, sz;
  float th,phi;
  float hx, hy, hz;
  //float norm;
  const int x = threadIdx.x % (flip_BlockSize_x);
  const int y = (threadIdx.x / flip_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % flip_BN) % flip_GridSize_x) * flip_BlockSize_x + x);
  const int ty =(blockIdx.x / flip_BN) * flip_SpinSize * flip_SpinSize_z +  3 * (((blockIdx.x % flip_BN) / flip_GridSize_x) * flip_BlockSize_y + y);
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
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, j+1, i)] + BWPxx * confx[flip_coo(k, j+1, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, ib)]\
     + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, j+1, i)] + BWPxy * confy[flip_coo(k, j+1, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, ib)]\
     + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, j+1, i)] + BWPxz * confz[flip_coo(k, j+1, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, ib)]\
		 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, j+1, i)] + BWPyx * confx[flip_coo(k, j+1, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, ib)]\
     + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, j+1, i)] + BWPyy * confy[flip_coo(k, j+1, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, ib)]\
     + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, j+1, i)] + BWPyz * confz[flip_coo(k, j+1, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, ib)]\
     + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, j+1, i)] + BWPzx * confx[flip_coo(k, j+1, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, ib)]\
     + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, j+1, i)] + BWPzy * confy[flip_coo(k, j+1, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, ib)]\
     + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, j+1, i)] + BWPzz * confz[flip_coo(k, j+1, i+1)] + BXMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, ib)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();
  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, j+1, i)] + BWPxx * confx[flip_coo(k, j+1, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, ib)]\
			 + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, j+1, i)] + BWPxy * confy[flip_coo(k, j+1, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, ib)]\
			 + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, j+1, i)] + BWPxz * confz[flip_coo(k, j+1, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, ib)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, j+1, i)] + BWPyx * confx[flip_coo(k, j+1, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, ib)]\
			 + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, j+1, i)] + BWPyy * confy[flip_coo(k, j+1, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, ib)]\
			 + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, j+1, i)] + BWPyz * confz[flip_coo(k, j+1, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, ib)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, j+1, i)] + BWPzx * confx[flip_coo(k, j+1, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, ib)]\
			 + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, j+1, i)] + BWPzy * confy[flip_coo(k, j+1, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, ib)]\
			 + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, j+1, i)] + BWPzz * confz[flip_coo(k, j+1, i+1)] + BXMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, ib)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();
	}
  //last layer
  k = SpinSize_z - 1;
	hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, j+1, i)] + BWPxx * confx[flip_coo(k, j+1, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, ib)]\
		 + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, j+1, i)] + BWPxy * confy[flip_coo(k, j+1, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, ib)]\
		 + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, j+1, i)] + BWPxz * confz[flip_coo(k, j+1, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, ib)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
	hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, j+1, i)] + BWPyx * confx[flip_coo(k, j+1, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, ib)]\
		 + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, j+1, i)] + BWPyy * confy[flip_coo(k, j+1, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, ib)]\
		 + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, j+1, i)] + BWPyz * confz[flip_coo(k, j+1, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, ib)]\
			+ BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
	hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, j+1, i)] + BWPzx * confx[flip_coo(k, j+1, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, ib)]\
		 + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, j+1, i)] + BWPzy * confy[flip_coo(k, j+1, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, ib)]\
		 + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, j+1, i)] + BWPzz * confz[flip_coo(k, j+1, i+1)] + BXMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, ib)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
	du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
		confx[flip_coo(k, j, i)] = sx;
		confy[flip_coo(k, j, i)] = sy;
		confz[flip_coo(k, j, i)] = sz;
	}
	else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
		confx[flip_coo(k, j, i)] = sx;
		confy[flip_coo(k, j, i)] = sy;
		confz[flip_coo(k, j, i)] = sz;
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
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
		 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();
  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();
	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
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
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
		 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();

	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}



__global__ void flip2_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float* invTs){
  //Energy variables
  //__shared__ unsigned rngShmem[1024];
  extern __shared__ unsigned rngShmem[];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float H = Hs[blockIdx.x / flip_BN];
  float invT = invTs[blockIdx.x / flip_BN];
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
  //...
  //0..
  //...
  i = tx;
  j = ty + 1;
  ib = (i + flip_SpinSize - 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, (j-1), i)] + BWMxx * confx[flip_coo(k, (j-1), ib)]\
     + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, (j-1), i)] + BWMxy * confy[flip_coo(k, (j-1), ib)]\
     + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, (j-1), i)] + BWMxz * confz[flip_coo(k, (j-1), ib)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, (j-1), i)] + BWMyx * confx[flip_coo(k, (j-1), ib)]\
     + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, (j-1), i)] + BWMyy * confy[flip_coo(k, (j-1), ib)]\
     + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, (j-1), i)] + BWMyz * confz[flip_coo(k, (j-1), ib)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, (j-1), i)] + BWMzx * confx[flip_coo(k, (j-1), ib)]\
     + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, (j-1), i)] + BWMzy * confy[flip_coo(k, (j-1), ib)]\
     + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, i+1)] + BWMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, (j-1), i)] + BWMzz * confz[flip_coo(k, (j-1), ib)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, (j-1), i)] + BWMxx * confx[flip_coo(k, (j-1), ib)]\
			 + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, (j-1), i)] + BWMxy * confy[flip_coo(k, (j-1), ib)]\
			 + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, (j-1), i)] + BWMxz * confz[flip_coo(k, (j-1), ib)]\
				 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, (j-1), i)] + BWMyx * confx[flip_coo(k, (j-1), ib)]\
			 + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, (j-1), i)] + BWMyy * confy[flip_coo(k, (j-1), ib)]\
			 + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, (j-1), i)] + BWMyz * confz[flip_coo(k, (j-1), ib)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, (j-1), i)] + BWMzx * confx[flip_coo(k, (j-1), ib)]\
			 + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, (j-1), i)] + BWMzy * confy[flip_coo(k, (j-1), ib)]\
			 + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, i+1)] + BWMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, (j-1), i)] + BWMzz * confz[flip_coo(k, (j-1), ib)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();

	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, (j-1), i)] + BWMxx * confx[flip_coo(k, (j-1), ib)]\
     + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, (j-1), i)] + BWMxy * confy[flip_coo(k, (j-1), ib)]\
     + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, (j-1), i)] + BWMxz * confz[flip_coo(k, (j-1), ib)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, (j-1), i)] + BWMyx * confx[flip_coo(k, (j-1), ib)]\
     + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, (j-1), i)] + BWMyy * confy[flip_coo(k, (j-1), ib)]\
     + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, (j-1), i)] + BWMyz * confz[flip_coo(k, (j-1), ib)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, (j-1), i)] + BWMzx * confx[flip_coo(k, (j-1), ib)]\
     + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, (j-1), i)] + BWMzy * confy[flip_coo(k, (j-1), ib)]\
     + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, i+1)] + BWMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, (j-1), i)] + BWMzz * confz[flip_coo(k, (j-1), ib)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  //----------Spin flip at the top and right corner of each thread sqare----------
  //.0.
  //...
  //...
  i = tx + 1;
  j = ty;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == 0)	jb = j + flip_SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, (j+1), i)] + BWPxx * confx[flip_coo(k, (j+1), ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, (j+1), i)] + BWPxy * confy[flip_coo(k, (j+1), ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, (j+1), i)] + BWPxz * confz[flip_coo(k, (j+1), ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, i-1)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, (j+1), i)] + BWPyx * confx[flip_coo(k, (j+1), ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, (j+1), i)] + BWPyy * confy[flip_coo(k, (j+1), ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, (j+1), i)] + BWPyz * confz[flip_coo(k, (j+1), ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, i-1)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, (j+1), i)] + BWPzx * confx[flip_coo(k, (j+1), ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, (j+1), i)] + BWPzy * confy[flip_coo(k, (j+1), ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, (j+1), i)] + BWPzz * confz[flip_coo(k, (j+1), ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, i-1)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, (j+1), i)] + BWPxx * confx[flip_coo(k, (j+1), ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, i-1)]\
			 + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, (j+1), i)] + BWPxy * confy[flip_coo(k, (j+1), ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, i-1)]\
			 + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, (j+1), i)] + BWPxz * confz[flip_coo(k, (j+1), ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, i-1)]\
				 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, (j+1), i)] + BWPyx * confx[flip_coo(k, (j+1), ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, i-1)]\
			 + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, (j+1), i)] + BWPyy * confy[flip_coo(k, (j+1), ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, i-1)]\
			 + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, (j+1), i)] + BWPyz * confz[flip_coo(k, (j+1), ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, i-1)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, (j+1), i)] + BWPzx * confx[flip_coo(k, (j+1), ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, i-1)]\
			 + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, (j+1), i)] + BWPzy * confy[flip_coo(k, (j+1), ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, i-1)]\
			 + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, (j+1), i)] + BWPzz * confz[flip_coo(k, (j+1), ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, i-1)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();

	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, (j+1), i)] + BWPxx * confx[flip_coo(k, (j+1), ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, (j+1), i)] + BWPxy * confy[flip_coo(k, (j+1), ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, (j+1), i)] + BWPxz * confz[flip_coo(k, (j+1), ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, i-1)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, (j+1), i)] + BWPyx * confx[flip_coo(k, (j+1), ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, (j+1), i)] + BWPyy * confy[flip_coo(k, (j+1), ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, (j+1), i)] + BWPyz * confz[flip_coo(k, (j+1), ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, i-1)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, (j+1), i)] + BWPzx * confx[flip_coo(k, (j+1), ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, (j+1), i)] + BWPzy * confy[flip_coo(k, (j+1), ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, (j+1), i)] + BWPzz * confz[flip_coo(k, (j+1), ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, i-1)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
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
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
				 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();

	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}


__global__ void flip3_TRI(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float* invTs){
  //Energy variables
  //__shared__ unsigned rngShmem[1024];
  extern __shared__ unsigned rngShmem[];
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(rngState, rngRegs, rngShmem);
  float H = Hs[blockIdx.x / flip_BN];
  float invT = invTs[blockIdx.x / flip_BN];
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
  //...
  //...
  //0..
  i = tx;
  j = ty + 2;
  ib = (i + flip_SpinSize - 1) % flip_SpinSize;
  if((j % flip_SpinSize) == flip_SpinSize - 1)	jb = j - flip_SpinSize + 1;
  else					jb = j + 1;
  //Spin flip!
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, (j-1), i)] + BWMxx * confx[flip_coo(k, (j-1), ib)]\
     + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, (j-1), i)] + BWMxy * confy[flip_coo(k, (j-1), ib)]\
     + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, (j-1), i)] + BWMxz * confz[flip_coo(k, (j-1), ib)]\
		 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, (j-1), i)] + BWMyx * confx[flip_coo(k, (j-1), ib)]\
     + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, (j-1), i)] + BWMyy * confy[flip_coo(k, (j-1), ib)]\
     + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, (j-1), i)] + BWMyz * confz[flip_coo(k, (j-1), ib)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, (j-1), i)] + BWMzx * confx[flip_coo(k, (j-1), ib)]\
     + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, (j-1), i)] + BWMzy * confy[flip_coo(k, (j-1), ib)]\
     + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, i+1)] + BWMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, (j-1), i)] + BWMzz * confz[flip_coo(k, (j-1), ib)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, (j-1), i)] + BWMxx * confx[flip_coo(k, (j-1), ib)]\
			 + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, (j-1), i)] + BWMxy * confy[flip_coo(k, (j-1), ib)]\
			 + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, (j-1), i)] + BWMxz * confz[flip_coo(k, (j-1), ib)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, (j-1), i)] + BWMyx * confx[flip_coo(k, (j-1), ib)]\
			 + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, (j-1), i)] + BWMyy * confy[flip_coo(k, (j-1), ib)]\
			 + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, (j-1), i)] + BWMyz * confz[flip_coo(k, (j-1), ib)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, (j-1), i)] + BWMzx * confx[flip_coo(k, (j-1), ib)]\
			 + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, (j-1), i)] + BWMzy * confy[flip_coo(k, (j-1), ib)]\
			 + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, i+1)] + BWMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, (j-1), i)] + BWMzz * confz[flip_coo(k, (j-1), ib)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();

	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, i+1)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, i+1)] + BXMxx * confx[flip_coo(k, j, ib)] + BYMxx * confx[flip_coo(k, (j-1), i)] + BWMxx * confx[flip_coo(k, (j-1), ib)]\
     + BXPxy * confy[flip_coo(k, j, i+1)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, i+1)] + BXMxy * confy[flip_coo(k, j, ib)] + BYMxy * confy[flip_coo(k, (j-1), i)] + BWMxy * confy[flip_coo(k, (j-1), ib)]\
     + BXPxz * confz[flip_coo(k, j, i+1)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, i+1)] + BXMxz * confz[flip_coo(k, j, ib)] + BYMxz * confz[flip_coo(k, (j-1), i)] + BWMxz * confz[flip_coo(k, (j-1), ib)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, i+1)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, i+1)] + BXMyx * confx[flip_coo(k, j, ib)] + BYMyx * confx[flip_coo(k, (j-1), i)] + BWMyx * confx[flip_coo(k, (j-1), ib)]\
     + BXPyy * confy[flip_coo(k, j, i+1)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, i+1)] + BXMyy * confy[flip_coo(k, j, ib)] + BYMyy * confy[flip_coo(k, (j-1), i)] + BWMyy * confy[flip_coo(k, (j-1), ib)]\
     + BXPyz * confz[flip_coo(k, j, i+1)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, i+1)] + BXMyz * confz[flip_coo(k, j, ib)] + BYMyz * confz[flip_coo(k, (j-1), i)] + BWMyz * confz[flip_coo(k, (j-1), ib)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, i+1)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, i+1)] + BXMzx * confx[flip_coo(k, j, ib)] + BYMzx * confx[flip_coo(k, (j-1), i)] + BWMzx * confx[flip_coo(k, (j-1), ib)]\
     + BXPzy * confy[flip_coo(k, j, i+1)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, i+1)] + BXMzy * confy[flip_coo(k, j, ib)] + BYMzy * confy[flip_coo(k, (j-1), i)] + BWMzy * confy[flip_coo(k, (j-1), ib)]\
     + BXPzz * confz[flip_coo(k, j, i+1)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, i+1)] + BWMzz * confz[flip_coo(k, j, ib)] + BYMzz * confz[flip_coo(k, (j-1), i)] + BWMzz * confz[flip_coo(k, (j-1), ib)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  //----------Spin flip at the top and right corner of each thread sqare----------
  //..0
  //...
  //...
  i = tx + 2;
  j = ty;
  ib = (i + 1) % flip_SpinSize;
  if((j % flip_SpinSize) == 0)	jb = j + flip_SpinSize - 1;
  else			jb = j - 1;
  //Spin flip!
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, (j+1), i)] + BWPxx * confx[flip_coo(k, (j+1), ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, (j+1), i)] + BWPxy * confy[flip_coo(k, (j+1), ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, (j+1), i)] + BWPxz * confz[flip_coo(k, (j+1), ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, i-1)]\
		 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, (j+1), i)] + BWPyx * confx[flip_coo(k, (j+1), ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, (j+1), i)] + BWPyy * confy[flip_coo(k, (j+1), ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, (j+1), i)] + BWPyz * confz[flip_coo(k, (j+1), ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, i-1)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, (j+1), i)] + BWPzx * confx[flip_coo(k, (j+1), ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, (j+1), i)] + BWPzy * confy[flip_coo(k, (j+1), ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, (j+1), i)] + BWPzz * confz[flip_coo(k, (j+1), ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, i-1)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, (j+1), i)] + BWPxx * confx[flip_coo(k, (j+1), ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, i-1)]\
			 + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, (j+1), i)] + BWPxy * confy[flip_coo(k, (j+1), ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, i-1)]\
			 + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, (j+1), i)] + BWPxz * confz[flip_coo(k, (j+1), ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, i-1)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, (j+1), i)] + BWPyx * confx[flip_coo(k, (j+1), ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, i-1)]\
			 + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, (j+1), i)] + BWPyy * confy[flip_coo(k, (j+1), ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, i-1)]\
			 + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, (j+1), i)] + BWPyz * confz[flip_coo(k, (j+1), ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, i-1)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, (j+1), i)] + BWPzx * confx[flip_coo(k, (j+1), ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, i-1)]\
			 + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, (j+1), i)] + BWPzy * confy[flip_coo(k, (j+1), ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, i-1)]\
			 + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, (j+1), i)] + BWPzz * confz[flip_coo(k, (j+1), ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, i-1)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();

	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, (j+1), i)] + BWPxx * confx[flip_coo(k, (j+1), ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, jb, i)] + BWMxx * confx[flip_coo(k, jb, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, (j+1), i)] + BWPxy * confy[flip_coo(k, (j+1), ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, jb, i)] + BWMxy * confy[flip_coo(k, jb, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, (j+1), i)] + BWPxz * confz[flip_coo(k, (j+1), ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, jb, i)] + BWMxz * confz[flip_coo(k, jb, i-1)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, (j+1), i)] + BWPyx * confx[flip_coo(k, (j+1), ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, jb, i)] + BWMyx * confx[flip_coo(k, jb, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, (j+1), i)] + BWPyy * confy[flip_coo(k, (j+1), ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, jb, i)] + BWMyy * confy[flip_coo(k, jb, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, (j+1), i)] + BWPyz * confz[flip_coo(k, (j+1), ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, jb, i)] + BWMyz * confz[flip_coo(k, jb, i-1)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, (j+1), i)] + BWPzx * confx[flip_coo(k, (j+1), ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, jb, i)] + BWMzx * confx[flip_coo(k, jb, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, (j+1), i)] + BWPzy * confy[flip_coo(k, (j+1), ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, jb, i)] + BWMzy * confy[flip_coo(k, jb, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, (j+1), i)] + BWPzz * confz[flip_coo(k, (j+1), ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, jb, i)] + BWMzz * confz[flip_coo(k, jb, i-1)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = - confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
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
  //first layer
	k = 0;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
		 + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZPzz * confz[coo(k+1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  for (k = 1;k < SpinSize_z - 1; k++){//middle layers
		hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
		hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
			 + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
		hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
			 + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
			 + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
			 + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
		du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}
		else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
			confx[flip_coo(k, j, i)] = sx;
			confy[flip_coo(k, j, i)] = sy;
			confz[flip_coo(k, j, i)] = sz;
		}

		__syncthreads();

	}
  //last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[flip_coo(k, j, ib)] + BYPxx * confx[flip_coo(k, jb, i)] + BWPxx * confx[flip_coo(k, jb, ib)] + BXMxx * confx[flip_coo(k, j, i-1)] + BYMxx * confx[flip_coo(k, j-1, i)] + BWMxx * confx[flip_coo(k, j-1, i-1)]\
     + BXPxy * confy[flip_coo(k, j, ib)] + BYPxy * confy[flip_coo(k, jb, i)] + BWPxy * confy[flip_coo(k, jb, ib)] + BXMxy * confy[flip_coo(k, j, i-1)] + BYMxy * confy[flip_coo(k, j-1, i)] + BWMxy * confy[flip_coo(k, j-1, i-1)]\
     + BXPxz * confz[flip_coo(k, j, ib)] + BYPxz * confz[flip_coo(k, jb, i)] + BWPxz * confz[flip_coo(k, jb, ib)] + BXMxz * confz[flip_coo(k, j, i-1)] + BYMxz * confz[flip_coo(k, j-1, i)] + BWMxz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[flip_coo(k, j, ib)] + BYPyx * confx[flip_coo(k, jb, i)] + BWPyx * confx[flip_coo(k, jb, ib)] + BXMyx * confx[flip_coo(k, j, i-1)] + BYMyx * confx[flip_coo(k, j-1, i)] + BWMyx * confx[flip_coo(k, j-1, i-1)]\
     + BXPyy * confy[flip_coo(k, j, ib)] + BYPyy * confy[flip_coo(k, jb, i)] + BWPyy * confy[flip_coo(k, jb, ib)] + BXMyy * confy[flip_coo(k, j, i-1)] + BYMyy * confy[flip_coo(k, j-1, i)] + BWMyy * confy[flip_coo(k, j-1, i-1)]\
     + BXPyz * confz[flip_coo(k, j, ib)] + BYPyz * confz[flip_coo(k, jb, i)] + BWPyz * confz[flip_coo(k, jb, ib)] + BXMyz * confz[flip_coo(k, j, i-1)] + BYMyz * confz[flip_coo(k, j-1, i)] + BWMyz * confz[flip_coo(k, j-1, i-1)]\
		 + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[flip_coo(k, j, ib)] + BYPzx * confx[flip_coo(k, jb, i)] + BWPzx * confx[flip_coo(k, jb, ib)] + BXMzx * confx[flip_coo(k, j, i-1)] + BYMzx * confx[flip_coo(k, j-1, i)] + BWMzx * confx[flip_coo(k, j-1, i-1)]\
     + BXPzy * confy[flip_coo(k, j, ib)] + BYPzy * confy[flip_coo(k, jb, i)] + BWPzy * confy[flip_coo(k, jb, ib)] + BXMzy * confy[flip_coo(k, j, i-1)] + BYMzy * confy[flip_coo(k, j-1, i)] + BWMzy * confy[flip_coo(k, j-1, i-1)]\
     + BXPzz * confz[flip_coo(k, j, ib)] + BYPzz * confz[flip_coo(k, jb, i)] + BWPzz * confz[flip_coo(k, jb, ib)] + BXMzz * confz[flip_coo(k, j, i-1)] + BYMzz * confz[flip_coo(k, j-1, i)] + BWMzz * confz[flip_coo(k, j-1, i-1)] + H\
		 + BZMzz * confz[coo(k-1, j, i)];
  du = -confx[flip_coo(k, j, i)] * hx - confy[flip_coo(k, j, i)] * hy - confz[flip_coo(k, j, i)] * hz + flip_A * confz[flip_coo(k, j, i)] * confz[flip_coo(k, j, i)];
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
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[flip_coo(k, j, i)] = sx;
    confy[flip_coo(k, j, i)] = sy;
    confz[flip_coo(k, j, i)] = sz;
  }

  __syncthreads();

  //Load random number back to global memory
  WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}
#endif
