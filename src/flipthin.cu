#ifndef UPDATE_H
#define UPDATE_H
#define "update.cuh"
#endif
__global__ void flipTLBRthin(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT){
	//Energy variables
	__shared__ unsigned rngShmem[BlockSize_x * BlockSize_y * 4];
	unsigned rngRegs[WarpStandard_REG_COUNT];
	WarpStandard_LoadState(rngState, rngRegs, rngShmem);
	float H = Hs[blockIdx.x / BN];
	unsigned int r;
	float du;	//-dE
	float sx, sy, sz;
	float th, phi;
	float hx, hy, hz;
	//float norm;
	const int x = threadIdx.x % (BlockSize_x);
	const int y = (threadIdx.x / BlockSize_x);
	const int tx = 2 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
	const int ty =(blockIdx.x / BN) * SpinSize * SpinSize_z + 2 * (((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
	int i, j, ib, jb, k;
	//----------Spin flip at the top and left corner of each thread sqare----------
	i = tx;
  j = ty;
  ib = (i + SpinSize - 1) % SpinSize;
	if((j % SpinSize) == 0)	jb = j + SpinSize - 1;
	else			jb = j - 1;
	//Spin flip!
	//first layer
	k = 0;
  hx = BXPxx * confx[coo(k, j, i+1)] + BYPxx * confx[coo(k, j+1, i)] + BXMxx * confx[coo(k, j, ib )] + BYMxx * confx[coo(k, jb , i)]\
     + BXPxy * confy[coo(k, j, i+1)] + BYPxy * confy[coo(k, j+1, i)] + BXMxy * confy[coo(k, j, ib )] + BYMxy * confy[coo(k, jb , i)]\
     + BXPxz * confz[coo(k, j, i+1)] + BYPxz * confz[coo(k, j+1, i)] + BXMxz * confz[coo(k, j, ib )] + BYMxz * confz[coo(k, jb , i)]\
     + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[coo(k, j, i+1)] + BYPyx * confx[coo(k, j+1, i)] + BXMyx * confx[coo(k, j, ib )] + BYMyx * confx[coo(k, jb , i)]\
     + BXPyy * confy[coo(k, j, i+1)] + BYPyy * confy[coo(k, j+1, i)] + BXMyy * confy[coo(k, j, ib )] + BYMyy * confy[coo(k, jb , i)]\
     + BXPyz * confz[coo(k, j, i+1)] + BYPyz * confz[coo(k, j+1, i)] + BXMyz * confz[coo(k, j, ib )] + BYMyz * confz[coo(k, jb , i)]\
     + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[coo(k, j, i+1)] + BYPzx * confx[coo(k, j+1, i)] + BXMzx * confx[coo(k, j, ib )] + BYMzx * confx[coo(k, jb , i)]\
     + BXPzy * confy[coo(k, j, i+1)] + BYPzy * confy[coo(k, j+1, i)] + BXMzy * confy[coo(k, j, ib )] + BYMzy * confy[coo(k, jb , i)]\
     + BXPzz * confz[coo(k, j, i+1)] + BYPzz * confz[coo(k, j+1, i)] + BXMzz * confz[coo(k, j, ib )] + BYMzz * confz[coo(k, jb , i)] + H\
     + BZPzz * confz[coo(k+1, j, i)];
  du =- confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
	}
	else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
	}

	__syncthreads();

	for (k = 1;k < SpinSize_z - 1; k++){//middle layers
    hx = BXPxx * confx[coo(k, j, i+1)] + BYPxx * confx[coo(k, j+1, i)] + BXMxx * confx[coo(k, j, ib )] + BYMxx * confx[coo(k, jb , i)]\
       + BXPxy * confy[coo(k, j, i+1)] + BYPxy * confy[coo(k, j+1, i)] + BXMxy * confy[coo(k, j, ib )] + BYMxy * confy[coo(k, jb , i)]\
       + BXPxz * confz[coo(k, j, i+1)] + BYPxz * confz[coo(k, j+1, i)] + BXMxz * confz[coo(k, j, ib )] + BYMxz * confz[coo(k, jb , i)]\
       + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
    hy = BXPyx * confx[coo(k, j, i+1)] + BYPyx * confx[coo(k, j+1, i)] + BXMyx * confx[coo(k, j, ib )] + BYMyx * confx[coo(k, jb , i)]\
       + BXPyy * confy[coo(k, j, i+1)] + BYPyy * confy[coo(k, j+1, i)] + BXMyy * confy[coo(k, j, ib )] + BYMyy * confy[coo(k, jb , i)]\
       + BXPyz * confz[coo(k, j, i+1)] + BYPyz * confz[coo(k, j+1, i)] + BXMyz * confz[coo(k, j, ib )] + BYMyz * confz[coo(k, jb , i)]\
       + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
    hz = BXPzx * confx[coo(k, j, i+1)] + BYPzx * confx[coo(k, j+1, i)] + BXMzx * confx[coo(k, j, ib )] + BYMzx * confx[coo(k, jb , i)]\
       + BXPzy * confy[coo(k, j, i+1)] + BYPzy * confy[coo(k, j+1, i)] + BXMzy * confy[coo(k, j, ib )] + BYMzy * confy[coo(k, jb , i)]\
       + BXPzz * confz[coo(k, j, i+1)] + BYPzz * confz[coo(k, j+1, i)] + BXMzz * confz[coo(k, j, ib )] + BYMzz * confz[coo(k, jb , i)] + H\
       + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
    du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }
    else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }

    __syncthreads();
  }//end middle layers

  //last layer
  k = SpinSize_z - 1;
  hx = BXPxx * confx[coo(k, j, i+1)] + BYPxx * confx[coo(k, j+1, i)] + BXMxx * confx[coo(k, j, ib )] + BYMxx * confx[coo(k, jb , i)]\
     + BXPxy * confy[coo(k, j, i+1)] + BYPxy * confy[coo(k, j+1, i)] + BXMxy * confy[coo(k, j, ib )] + BYMxy * confy[coo(k, jb , i)]\
     + BXPxz * confz[coo(k, j, i+1)] + BYPxz * confz[coo(k, j+1, i)] + BXMxz * confz[coo(k, j, ib )] + BYMxz * confz[coo(k, jb , i)]\
     + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[coo(k, j, i+1)] + BYPyx * confx[coo(k, j+1, i)] + BXMyx * confx[coo(k, j, ib )] + BYMyx * confx[coo(k, jb , i)]\
     + BXPyy * confy[coo(k, j, i+1)] + BYPyy * confy[coo(k, j+1, i)] + BXMyy * confy[coo(k, j, ib )] + BYMyy * confy[coo(k, jb , i)]\
     + BXPyz * confz[coo(k, j, i+1)] + BYPyz * confz[coo(k, j+1, i)] + BXMyz * confz[coo(k, j, ib )] + BYMyz * confz[coo(k, jb , i)]\
     + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[coo(k, j, i+1)] + BYPzx * confx[coo(k, j+1, i)] + BXMzx * confx[coo(k, j, ib )] + BYMzx * confx[coo(k, jb , i)]\
     + BXPzy * confy[coo(k, j, i+1)] + BYPzy * confy[coo(k, j+1, i)] + BXMzy * confy[coo(k, j, ib )] + BYMzy * confy[coo(k, jb , i)]\
     + BXPzz * confz[coo(k, j, i+1)] + BYPzz * confz[coo(k, j+1, i)] + BXMzz * confz[coo(k, j, ib )] + BYMzz * confz[coo(k, jb , i)] + H\
     + BZMzz * confz[coo(k-1, j, i)];
  du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
	}
	else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
	}

	__syncthreads();

	//----------Spin flip at the bottom and right corner of each thread sqare----------
	i = tx + 1;
	j = ty + 1;
	ib = (i + 1) % SpinSize;
	if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
	else					jb = j + 1;
	//Spin flip!
	//first layer
	k = 0;
  hx = BXPxx * confx[coo(k, j, ib)] + BYPxx * confx[coo(k, jb, i)] + BXMxx * confx[coo(k, j, i-1)] + BYMxx * confx[coo(k, j-1 , i)]\
     + BXPxy * confy[coo(k, j, ib)] + BYPxy * confy[coo(k, jb, i)] + BXMxy * confy[coo(k, j, i-1)] + BYMxy * confy[coo(k, j-1 , i)]\
     + BXPxz * confz[coo(k, j, ib)] + BYPxz * confz[coo(k, jb, i)] + BXMxz * confz[coo(k, j, i-1)] + BYMxz * confz[coo(k, j-1 , i)]\
     + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
  hy = BXPyx * confx[coo(k, j, ib)] + BYPyx * confx[coo(k, jb, i)] + BXMyx * confx[coo(k, j, i-1)] + BYMyx * confx[coo(k, j-1 , i)]\
     + BXPyy * confy[coo(k, j, ib)] + BYPyy * confy[coo(k, jb, i)] + BXMyy * confy[coo(k, j, i-1)] + BYMyy * confy[coo(k, j-1 , i)]\
     + BXPyz * confz[coo(k, j, ib)] + BYPyz * confz[coo(k, jb, i)] + BXMyz * confz[coo(k, j, i-1)] + BYMyz * confz[coo(k, j-1 , i)]\
     + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
  hz = BXPzx * confx[coo(k, j, ib)] + BYPzx * confx[coo(k, jb, i)] + BXMzx * confx[coo(k, j, i-1)] + BYMzx * confx[coo(k, j-1 , i)]\
     + BXPzy * confy[coo(k, j, ib)] + BYPzy * confy[coo(k, jb, i)] + BXMzy * confy[coo(k, j, i-1)] + BYMzy * confy[coo(k, j-1 , i)]\
     + BXPzz * confz[coo(k, j, ib)] + BYPzz * confz[coo(k, jb, i)] + BXMzz * confz[coo(k, j, i-1)] + BYMzz * confz[coo(k, j-1 , i)] + H\
     + BZPzz * confz[coo(k+1, j, i)];
  du =- confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
  }

  __syncthreads();

	for (k = 1;k < SpinSize_z - 1; k++){//middle layers
    hx = BXPxx * confx[coo(k, j, ib)] + BYPxx * confx[coo(k, jb, i)] + BXMxx * confx[coo(k, j, i-1)] + BYMxx * confx[coo(k, j-1 , i)]\
       + BXPxy * confy[coo(k, j, ib)] + BYPxy * confy[coo(k, jb, i)] + BXMxy * confy[coo(k, j, i-1)] + BYMxy * confy[coo(k, j-1 , i)]\
       + BXPxz * confz[coo(k, j, ib)] + BYPxz * confz[coo(k, jb, i)] + BXMxz * confz[coo(k, j, i-1)] + BYMxz * confz[coo(k, j-1 , i)]\
       + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
    hy = BXPyx * confx[coo(k, j, ib)] + BYPyx * confx[coo(k, jb, i)] + BXMyx * confx[coo(k, j, i-1)] + BYMyx * confx[coo(k, j-1 , i)]\
       + BXPyy * confy[coo(k, j, ib)] + BYPyy * confy[coo(k, jb, i)] + BXMyy * confy[coo(k, j, i-1)] + BYMyy * confy[coo(k, j-1 , i)]\
       + BXPyz * confz[coo(k, j, ib)] + BYPyz * confz[coo(k, jb, i)] + BXMyz * confz[coo(k, j, i-1)] + BYMyz * confz[coo(k, j-1 , i)]\
       + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
    hz = BXPzx * confx[coo(k, j, ib)] + BYPzx * confx[coo(k, jb, i)] + BXMzx * confx[coo(k, j, i-1)] + BYMzx * confx[coo(k, j-1 , i)]\
       + BXPzy * confy[coo(k, j, ib)] + BYPzy * confy[coo(k, jb, i)] + BXMzy * confy[coo(k, j, i-1)] + BYMzy * confy[coo(k, j-1 , i)]\
       + BXPzz * confz[coo(k, j, ib)] + BYPzz * confz[coo(k, jb, i)] + BXMzz * confz[coo(k, j, i-1)] + BYMzz * confz[coo(k, j-1 , i)] + H\
       + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
    du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }
    else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }

    __syncthreads();
  }//end middle layers

	//last layer
	k = SpinSize_z - 1;
  hx = BXPxx * confx[coo(k, j, ib)] + BYPxx * confx[coo(k, jb, i)] + BXMxx * confx[coo(k, j, i-1)] + BYMxx * confx[coo(k, j-1 , i)]\
     + BXPxy * confy[coo(k, j, ib)] + BYPxy * confy[coo(k, jb, i)] + BXMxy * confy[coo(k, j, i-1)] + BYMxy * confy[coo(k, j-1 , i)]\
     + BXPxz * confz[coo(k, j, ib)] + BYPxz * confz[coo(k, jb, i)] + BXMxz * confz[coo(k, j, i-1)] + BYMxz * confz[coo(k, j-1 , i)]\
     + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
  hy = BXPyx * confx[coo(k, j, ib)] + BYPyx * confx[coo(k, jb, i)] + BXMyx * confx[coo(k, j, i-1)] + BYMyx * confx[coo(k, j-1 , i)]\
     + BXPyy * confy[coo(k, j, ib)] + BYPyy * confy[coo(k, jb, i)] + BXMyy * confy[coo(k, j, i-1)] + BYMyy * confy[coo(k, j-1 , i)]\
     + BXPyz * confz[coo(k, j, ib)] + BYPyz * confz[coo(k, jb, i)] + BXMyz * confz[coo(k, j, i-1)] + BYMyz * confz[coo(k, j-1 , i)]\
     + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
  hz = BXPzx * confx[coo(k, j, ib)] + BYPzx * confx[coo(k, jb, i)] + BXMzx * confx[coo(k, j, i-1)] + BYMzx * confx[coo(k, j-1 , i)]\
     + BXPzy * confy[coo(k, j, ib)] + BYPzy * confy[coo(k, jb, i)] + BXMzy * confy[coo(k, j, i-1)] + BYMzy * confy[coo(k, j-1 , i)]\
     + BXPzz * confz[coo(k, j, ib)] + BYPzz * confz[coo(k, jb, i)] + BXMzz * confz[coo(k, j, i-1)] + BYMzz * confz[coo(k, j-1 , i)] + H\
     + BZMzz * confz[coo(k-1, j, i)];
  du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
		confx[coo(k, j, i)] = sx;
		confy[coo(k, j, i)] = sy;
		confz[coo(k, j, i)] = sz;
  }

  __syncthreads();


	//Load random number back to global memory
	WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}

__global__ void flipBLTRthin(float *confx, float *confy, float *confz, unsigned int *rngState, float* Hs, float invT){
	//Energy variables
	__shared__ unsigned rngShmem[BlockSize_x * BlockSize_y * 4];
	unsigned rngRegs[WarpStandard_REG_COUNT];
	WarpStandard_LoadState(rngState, rngRegs, rngShmem);
	float H = Hs[blockIdx.x / BN];
	unsigned int r;
	float du;	//-dE
	float sx, sy, sz;
	float th,phi;
	float hx, hy, hz;
	//float norm;
	const int x = threadIdx.x % (BlockSize_x);
	const int y = (threadIdx.x / BlockSize_x);// % BlockSize_y;
	const int tx = 2 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
	const int ty = (blockIdx.x / BN) * SpinSize * SpinSize_z + 2 * (((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
	int i, j, ib, jb, k;
	//----------Spin flip at the bottom and left corner of each thread sqare----------
	i = tx;
	j = ty + 1;
	ib = (i + SpinSize - 1) % SpinSize;
	if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
	else					jb = j + 1;
	//Spin flip!
	//first layer
	k = 0;
    hx = BXPxx * confx[coo(k, j, i+1)] + BYPxx * confx[coo(k, jb, i)] + BXMxx * confx[coo(k, j, ib )] + BYMxx * confx[coo(k, j-1 , i)]\
       + BXPxy * confy[coo(k, j, i+1)] + BYPxy * confy[coo(k, jb, i)] + BXMxy * confy[coo(k, j, ib )] + BYMxy * confy[coo(k, j-1 , i)]\
       + BXPxz * confz[coo(k, j, i+1)] + BYPxz * confz[coo(k, jb, i)] + BXMxz * confz[coo(k, j, ib )] + BYMxz * confz[coo(k, j-1 , i)]\
       + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
    hy = BXPyx * confx[coo(k, j, i+1)] + BYPyx * confx[coo(k, jb, i)] + BXMyx * confx[coo(k, j, ib )] + BYMyx * confx[coo(k, j-1 , i)]\
       + BXPyy * confy[coo(k, j, i+1)] + BYPyy * confy[coo(k, jb, i)] + BXMyy * confy[coo(k, j, ib )] + BYMyy * confy[coo(k, j-1 , i)]\
       + BXPyz * confz[coo(k, j, i+1)] + BYPyz * confz[coo(k, jb, i)] + BXMyz * confz[coo(k, j, ib )] + BYMyz * confz[coo(k, j-1 , i)]\
       + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
    hz = BXPzx * confx[coo(k, j, i+1)] + BYPzx * confx[coo(k, jb, i)] + BXMzx * confx[coo(k, j, ib )] + BYMzx * confx[coo(k, j-1 , i)]\
       + BXPzy * confy[coo(k, j, i+1)] + BYPzy * confy[coo(k, jb, i)] + BXMzy * confy[coo(k, j, ib )] + BYMzy * confy[coo(k, j-1 , i)]\
       + BXPzz * confz[coo(k, j, i+1)] + BYPzz * confz[coo(k, jb, i)] + BXMzz * confz[coo(k, j, ib )] + BYMzz * confz[coo(k, j-1 , i)] + H\
       + BZPzz * confz[coo(k+1, j, i)];
  du =- confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }

  __syncthreads();

	for (k = 1;k < SpinSize_z - 1; k++){//middle layers
    hx = BXPxx * confx[coo(k, j, i+1)] + BYPxx * confx[coo(k, jb, i)] + BXMxx * confx[coo(k, j, ib )] + BYMxx * confx[coo(k, j-1 , i)]\
       + BXPxy * confy[coo(k, j, i+1)] + BYPxy * confy[coo(k, jb, i)] + BXMxy * confy[coo(k, j, ib )] + BYMxy * confy[coo(k, j-1 , i)]\
       + BXPxz * confz[coo(k, j, i+1)] + BYPxz * confz[coo(k, jb, i)] + BXMxz * confz[coo(k, j, ib )] + BYMxz * confz[coo(k, j-1 , i)]\
       + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
    hy = BXPyx * confx[coo(k, j, i+1)] + BYPyx * confx[coo(k, jb, i)] + BXMyx * confx[coo(k, j, ib )] + BYMyx * confx[coo(k, j-1 , i)]\
       + BXPyy * confy[coo(k, j, i+1)] + BYPyy * confy[coo(k, jb, i)] + BXMyy * confy[coo(k, j, ib )] + BYMyy * confy[coo(k, j-1 , i)]\
       + BXPyz * confz[coo(k, j, i+1)] + BYPyz * confz[coo(k, jb, i)] + BXMyz * confz[coo(k, j, ib )] + BYMyz * confz[coo(k, j-1 , i)]\
       + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
    hz = BXPzx * confx[coo(k, j, i+1)] + BYPzx * confx[coo(k, jb, i)] + BXMzx * confx[coo(k, j, ib )] + BYMzx * confx[coo(k, j-1 , i)]\
       + BXPzy * confy[coo(k, j, i+1)] + BYPzy * confy[coo(k, jb, i)] + BXMzy * confy[coo(k, j, ib )] + BYMzy * confy[coo(k, j-1 , i)]\
       + BXPzz * confz[coo(k, j, i+1)] + BYPzz * confz[coo(k, jb, i)] + BXMzz * confz[coo(k, j, ib )] + BYMzz * confz[coo(k, j-1 , i)] + H\
       + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
    du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }
    else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }

    __syncthreads();
  }//end middle layers
  //last layer
  k = SpinSize_z - 1;
    hx = BXPxx * confx[coo(k, j, i+1)] + BYPxx * confx[coo(k, jb, i)] + BXMxx * confx[coo(k, j, ib )] + BYMxx * confx[coo(k, j-1 , i)]\
       + BXPxy * confy[coo(k, j, i+1)] + BYPxy * confy[coo(k, jb, i)] + BXMxy * confy[coo(k, j, ib )] + BYMxy * confy[coo(k, j-1 , i)]\
       + BXPxz * confz[coo(k, j, i+1)] + BYPxz * confz[coo(k, jb, i)] + BXMxz * confz[coo(k, j, ib )] + BYMxz * confz[coo(k, j-1 , i)]\
       + BZMxy * confy[coo(k-1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
    hy = BXPyx * confx[coo(k, j, i+1)] + BYPyx * confx[coo(k, jb, i)] + BXMyx * confx[coo(k, j, ib )] + BYMyx * confx[coo(k, j-1 , i)]\
       + BXPyy * confy[coo(k, j, i+1)] + BYPyy * confy[coo(k, jb, i)] + BXMyy * confy[coo(k, j, ib )] + BYMyy * confy[coo(k, j-1 , i)]\
       + BXPyz * confz[coo(k, j, i+1)] + BYPyz * confz[coo(k, jb, i)] + BXMyz * confz[coo(k, j, ib )] + BYMyz * confz[coo(k, j-1 , i)]\
       + BZMyx * confx[coo(k-1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
    hz = BXPzx * confx[coo(k, j, i+1)] + BYPzx * confx[coo(k, jb, i)] + BXMzx * confx[coo(k, j, ib )] + BYMzx * confx[coo(k, j-1 , i)]\
       + BXPzy * confy[coo(k, j, i+1)] + BYPzy * confy[coo(k, jb, i)] + BXMzy * confy[coo(k, j, ib )] + BYMzy * confy[coo(k, j-1 , i)]\
       + BXPzz * confz[coo(k, j, i+1)] + BYPzz * confz[coo(k, jb, i)] + BXMzz * confz[coo(k, j, ib )] + BYMzz * confz[coo(k, j-1 , i)] + H\
       + BZMzz * confz[coo(k-1, j, i)];
  du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }

  __syncthreads();

	//----------Spin flip at the top and right corner of each thread sqare----------
	i = tx + 1;
	j = ty;
	ib = (i + 1) % SpinSize;
	if((j % SpinSize) == 0)	jb = j + SpinSize - 1;
	else			jb = j - 1;
	//Spin flip!
	k = 0;
    hx = BXPxx * confx[coo(k, j, ib)] + BYPxx * confx[coo(k, j+1, i)] + BXMxx * confx[coo(k, j, i-1 )] + BYMxx * confx[coo(k, jb , i)]\
       + BXPxy * confy[coo(k, j, ib)] + BYPxy * confy[coo(k, j+1, i)] + BXMxy * confy[coo(k, j, i-1 )] + BYMxy * confy[coo(k, jb , i)]\
       + BXPxz * confz[coo(k, j, ib)] + BYPxz * confz[coo(k, j+1, i)] + BXMxz * confz[coo(k, j, i-1 )] + BYMxz * confz[coo(k, jb , i)]\
       + BZPxy * confy[coo(k+1, j, i)] + BZPxx * confx[coo(k+1, j, i)];
    hy = BXPyx * confx[coo(k, j, ib)] + BYPyx * confx[coo(k, j+1, i)] + BXMyx * confx[coo(k, j, i-1 )] + BYMyx * confx[coo(k, jb , i)]\
       + BXPyy * confy[coo(k, j, ib)] + BYPyy * confy[coo(k, j+1, i)] + BXMyy * confy[coo(k, j, i-1 )] + BYMyy * confy[coo(k, jb , i)]\
       + BXPyz * confz[coo(k, j, ib)] + BYPyz * confz[coo(k, j+1, i)] + BXMyz * confz[coo(k, j, i-1 )] + BYMyz * confz[coo(k, jb , i)]\
       + BZPyx * confx[coo(k+1, j, i)] + BZPyy * confy[coo(k+1, j, i)];
    hz = BXPzx * confx[coo(k, j, ib)] + BYPzx * confx[coo(k, j+1, i)] + BXMzx * confx[coo(k, j, i-1 )] + BYMzx * confx[coo(k, jb , i)]\
       + BXPzy * confy[coo(k, j, ib)] + BYPzy * confy[coo(k, j+1, i)] + BXMzy * confy[coo(k, j, i-1 )] + BYMzy * confy[coo(k, jb , i)]\
       + BXPzz * confz[coo(k, j, ib)] + BYPzz * confz[coo(k, j+1, i)] + BXMzz * confz[coo(k, j, i-1 )] + BYMzz * confz[coo(k, jb , i)] + H\
       + BZPzz * confz[coo(k+1, j, i)];
  du =- confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }

  __syncthreads();
	for (k = 1;k < SpinSize_z - 1; k++){//middle layers
    hx = BXPxx * confx[coo(k, j, ib)] + BYPxx * confx[coo(k, j+1, i)] + BXMxx * confx[coo(k, j, i-1 )] + BYMxx * confx[coo(k, jb , i)]\
       + BXPxy * confy[coo(k, j, ib)] + BYPxy * confy[coo(k, j+1, i)] + BXMxy * confy[coo(k, j, i-1 )] + BYMxy * confy[coo(k, jb , i)]\
       + BXPxz * confz[coo(k, j, ib)] + BYPxz * confz[coo(k, j+1, i)] + BXMxz * confz[coo(k, j, i-1 )] + BYMxz * confz[coo(k, jb , i)]\
       + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
    hy = BXPyx * confx[coo(k, j, ib)] + BYPyx * confx[coo(k, j+1, i)] + BXMyx * confx[coo(k, j, i-1 )] + BYMyx * confx[coo(k, jb , i)]\
       + BXPyy * confy[coo(k, j, ib)] + BYPyy * confy[coo(k, j+1, i)] + BXMyy * confy[coo(k, j, i-1 )] + BYMyy * confy[coo(k, jb , i)]\
       + BXPyz * confz[coo(k, j, ib)] + BYPyz * confz[coo(k, j+1, i)] + BXMyz * confz[coo(k, j, i-1 )] + BYMyz * confz[coo(k, jb , i)]\
       + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
    hz = BXPzx * confx[coo(k, j, ib)] + BYPzx * confx[coo(k, j+1, i)] + BXMzx * confx[coo(k, j, i-1 )] + BYMzx * confx[coo(k, jb , i)]\
       + BXPzy * confy[coo(k, j, ib)] + BYPzy * confy[coo(k, j+1, i)] + BXMzy * confy[coo(k, j, i-1 )] + BYMzy * confy[coo(k, jb , i)]\
       + BXPzz * confz[coo(k, j, ib)] + BYPzz * confz[coo(k, j+1, i)] + BXMzz * confz[coo(k, j, i-1 )] + BYMzz * confz[coo(k, jb , i)] + H\
       + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
    du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }
    else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
      confx[coo(k, j, i)] = sx;
      confy[coo(k, j, i)] = sy;
      confz[coo(k, j, i)] = sz;
    }

    __syncthreads();
  }//end middle layers
  //last layer
  k = SpinSize_z - 1;
    hx = BXPxx * confx[coo(k, j, ib)] + BYPxx * confx[coo(k, j+1, i)] + BXMxx * confx[coo(k, j, i-1 )] + BYMxx * confx[coo(k, jb , i)]\
       + BXPxy * confy[coo(k, j, ib)] + BYPxy * confy[coo(k, j+1, i)] + BXMxy * confy[coo(k, j, i-1 )] + BYMxy * confy[coo(k, jb , i)]\
       + BXPxz * confz[coo(k, j, ib)] + BYPxz * confz[coo(k, j+1, i)] + BXMxz * confz[coo(k, j, i-1 )] + BYMxz * confz[coo(k, jb , i)]\
       + BZPxy * confy[coo(k+1, j, i)] + BZMxy * confy[coo(k-1, j, i)] + BZPxx * confx[coo(k+1, j, i)] + BZMxx * confx[coo(k-1, j, i)];
    hy = BXPyx * confx[coo(k, j, ib)] + BYPyx * confx[coo(k, j+1, i)] + BXMyx * confx[coo(k, j, i-1 )] + BYMyx * confx[coo(k, jb , i)]\
       + BXPyy * confy[coo(k, j, ib)] + BYPyy * confy[coo(k, j+1, i)] + BXMyy * confy[coo(k, j, i-1 )] + BYMyy * confy[coo(k, jb , i)]\
       + BXPyz * confz[coo(k, j, ib)] + BYPyz * confz[coo(k, j+1, i)] + BXMyz * confz[coo(k, j, i-1 )] + BYMyz * confz[coo(k, jb , i)]\
       + BZPyx * confx[coo(k+1, j, i)] + BZMyx * confx[coo(k-1, j, i)] + BZPyy * confy[coo(k+1, j, i)] + BZMyy * confy[coo(k-1, j, i)];
    hz = BXPzx * confx[coo(k, j, ib)] + BYPzx * confx[coo(k, j+1, i)] + BXMzx * confx[coo(k, j, i-1 )] + BYMzx * confx[coo(k, jb , i)]\
       + BXPzy * confy[coo(k, j, ib)] + BYPzy * confy[coo(k, j+1, i)] + BXMzy * confy[coo(k, j, i-1 )] + BYMzy * confy[coo(k, jb , i)]\
       + BXPzz * confz[coo(k, j, ib)] + BYPzz * confz[coo(k, j+1, i)] + BXMzz * confz[coo(k, j, i-1 )] + BYMzz * confz[coo(k, jb , i)] + H\
       + BZPzz * confz[coo(k+1, j, i)] + BZMzz * confz[coo(k-1, j, i)];
    du -= confx[coo(k, j, i)] * hx - confy[coo(k, j, i)] * hy - confz[coo(k, j, i)] * hz + A * confz[coo(k, j, i)] * confz[coo(k, j, i)];
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
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }
  else if((unsigned int)(exp(du * invT) * UINT_MAX) > r){
    confx[coo(k, j, i)] = sx;
    confy[coo(k, j, i)] = sy;
    confz[coo(k, j, i)] = sz;
  }

  __syncthreads();

	//Load random number back to global memory
	WarpStandard_SaveState(rngRegs, rngShmem, rngState);
}
