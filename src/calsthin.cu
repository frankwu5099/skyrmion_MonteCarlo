#ifdef THIN
#include "measurements.cuh"
__global__ void calthin(float *confx, float *confy, float *confz, double *out){
	//Energy variables
	extern __shared__ double sD[];
	const int x = threadIdx.x % (BlockSize_x);
	const int y = (threadIdx.x / BlockSize_x);
	const int tx = 2 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
	const int ty =(blockIdx.x / BN) * SpinSize * SpinSize_z +  2 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
	const int txp = tx + 1;
	const int typ = ty + 1;
	//const int ty = 2 * ((blockIdx.x / BN) * SpinSize + ((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
	const int dataoff = (blockIdx.x / BN) * MEASURE_NUM * BN;
	int bx, by, z;
	//-----Calculate the energy of each spin pairs in the system-----
	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.

	bx = (tx + SpinSize - 1) % SpinSize;
	if((ty % SpinSize) == 0)	by = ty + SpinSize - 1;
	else				by = ty - 1;
	//Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.
	z = 0;
  //Top-left corner
  sD[y][x] = -confx[coo(z, ty, tx)] * (BXMxx * confx[coo(z, ty, bx)] + BYMxx * confx[coo(z, by, tx)])\
             +confx[coo(z, ty, tx)] * (BXMxy * confy[coo(z, ty, bx)] + BYMxy * confy[coo(z, by, tx)])\
             +confx[coo(z, ty, tx)] * (BXMxz * confz[coo(z, ty, bx)] + BYMxz * confz[coo(z, by, tx)])\
             +confy[coo(z, ty, tx)] * (BXMyx * confx[coo(z, ty, bx)] + BYMyx * confx[coo(z, by, tx)])\
             +confy[coo(z, ty, tx)] * (BXMyy * confy[coo(z, ty, bx)] + BYMyy * confy[coo(z, by, tx)])\
             +confy[coo(z, ty, tx)] * (BXMyz * confz[coo(z, ty, bx)] + BYMyz * confz[coo(z, by, tx)])\
             +confz[coo(z, ty, tx)] * (BXMzx * confx[coo(z, ty, bx)] + BYMzx * confx[coo(z, by, tx)])\
             +confz[coo(z, ty, tx)] * (BXMzy * confy[coo(z, ty, bx)] + BYMzy * confy[coo(z, by, tx)])\
             +confz[coo(z, ty, tx)] * (BXMzz * confz[coo(z, ty, bx)] + BYMzz * confz[coo(z, by, tx)] - A * confz[coo(z, ty, tx)]);
  //Bottom-left corner
  sD[y][x] -= confx[coo(z, ty+1, tx)] * (BXMxx * confx[coo(z, ty+1, bx)] + BYMxx * confx[coo(z, ty, tx)])\
             +confx[coo(z, ty+1, tx)] * (BXMxy * confy[coo(z, ty+1, bx)] + BYMxy * confy[coo(z, ty, tx)])\
             +confx[coo(z, ty+1, tx)] * (BXMxz * confz[coo(z, ty+1, bx)] + BYMxz * confz[coo(z, ty, tx)])\
             +confy[coo(z, ty+1, tx)] * (BXMyx * confx[coo(z, ty+1, bx)] + BYMyx * confx[coo(z, ty, tx)])\
             +confy[coo(z, ty+1, tx)] * (BXMyy * confy[coo(z, ty+1, bx)] + BYMyy * confy[coo(z, ty, tx)])\
             +confy[coo(z, ty+1, tx)] * (BXMyz * confz[coo(z, ty+1, bx)] + BYMyz * confz[coo(z, ty, tx)])\
             +confz[coo(z, ty+1, tx)] * (BXMzx * confx[coo(z, ty+1, bx)] + BYMzx * confx[coo(z, ty, tx)])\
             +confz[coo(z, ty+1, tx)] * (BXMzy * confy[coo(z, ty+1, bx)] + BYMzy * confy[coo(z, ty, tx)])\
             +confz[coo(z, ty+1, tx)] * (BXMzz * confz[coo(z, ty+1, bx)] + BYMzz * confz[coo(z, ty, tx)] - A * confz[coo(z, ty+1, tx)]);
  //Top-right corner
  sD[y][x] -= confx[coo(z, ty, tx+1)] * (BXMxx * confx[coo(z, ty, tx)] + BYMxx * confx[coo(z, by, tx+1)])\
             +confx[coo(z, ty, tx+1)] * (BXMxy * confy[coo(z, ty, tx)] + BYMxy * confy[coo(z, by, tx+1)])\
             +confx[coo(z, ty, tx+1)] * (BXMxz * confz[coo(z, ty, tx)] + BYMxz * confz[coo(z, by, tx+1)])\
             +confy[coo(z, ty, tx+1)] * (BXMyx * confx[coo(z, ty, tx)] + BYMyx * confx[coo(z, by, tx+1)])\
             +confy[coo(z, ty, tx+1)] * (BXMyy * confy[coo(z, ty, tx)] + BYMyy * confy[coo(z, by, tx+1)])\
             +confy[coo(z, ty, tx+1)] * (BXMyz * confz[coo(z, ty, tx)] + BYMyz * confz[coo(z, by, tx+1)])\
             +confz[coo(z, ty, tx+1)] * (BXMzx * confx[coo(z, ty, tx)] + BYMzx * confx[coo(z, by, tx+1)])\
             +confz[coo(z, ty, tx+1)] * (BXMzy * confy[coo(z, ty, tx)] + BYMzy * confy[coo(z, by, tx+1)])\
             +confz[coo(z, ty, tx+1)] * (BXMzz * confz[coo(z, ty, tx)] + BYMzz * confz[coo(z, by, tx+1)] - A * confz[coo(z, ty, tx+1)]);
  //Bottom-right corner
  sD[y][x] -= confx[coo(z, ty+1, tx+1)] * (BXMxx * confx[coo(z, ty+1, tx)] + BYMxx * confx[coo(z, ty, tx+1)])\
             +confx[coo(z, ty+1, tx+1)] * (BXMxy * confy[coo(z, ty+1, tx)] + BYMxy * confy[coo(z, ty, tx+1)])\
             +confx[coo(z, ty+1, tx+1)] * (BXMxz * confz[coo(z, ty+1, tx)] + BYMxz * confz[coo(z, ty, tx+1)])\
             +confy[coo(z, ty+1, tx+1)] * (BXMyx * confx[coo(z, ty+1, tx)] + BYMyx * confx[coo(z, ty, tx+1)])\
             +confy[coo(z, ty+1, tx+1)] * (BXMyy * confy[coo(z, ty+1, tx)] + BYMyy * confy[coo(z, ty, tx+1)])\
             +confy[coo(z, ty+1, tx+1)] * (BXMyz * confz[coo(z, ty+1, tx)] + BYMyz * confz[coo(z, ty, tx+1)])\
             +confz[coo(z, ty+1, tx+1)] * (BXMzx * confx[coo(z, ty+1, tx)] + BYMzx * confx[coo(z, ty, tx+1)])\
             +confz[coo(z, ty+1, tx+1)] * (BXMzy * confy[coo(z, ty+1, tx)] + BYMzy * confy[coo(z, ty, tx+1)])\
             +confz[coo(z, ty+1, tx+1)] * (BXMzz * confz[coo(z, ty+1, tx)] + BYMzz * confz[coo(z, ty, tx+1)] - A * confz[coo(z, ty+1, tx+1)]);

  for (z = 1; z < SpinSize_z; z++){
    //Top-left corner
    sD[y][x] -= confx[coo(z, ty, tx)] * (BXMxx * confx[coo(z, ty, bx)] + BYMxx * confx[coo(z, by, tx)] + BZMxx * confx[coo(z-1, ty, tx)])\
               +confx[coo(z, ty, tx)] * (BXMxy * confy[coo(z, ty, bx)] + BYMxy * confy[coo(z, by, tx)] + BZMxy * confy[coo(z-1, ty, tx)])\
               +confx[coo(z, ty, tx)] * (BXMxz * confz[coo(z, ty, bx)] + BYMxz * confz[coo(z, by, tx)])\
               +confy[coo(z, ty, tx)] * (BXMyx * confx[coo(z, ty, bx)] + BYMyx * confx[coo(z, by, tx)] + BZMyx * confx[coo(z-1, ty, tx)])\
               +confy[coo(z, ty, tx)] * (BXMyy * confy[coo(z, ty, bx)] + BYMyy * confy[coo(z, by, tx)] + BZMyy * confy[coo(z-1, ty, tx)])\
               +confy[coo(z, ty, tx)] * (BXMyz * confz[coo(z, ty, bx)] + BYMyz * confz[coo(z, by, tx)])\
               +confz[coo(z, ty, tx)] * (BXMzx * confx[coo(z, ty, bx)] + BYMzx * confx[coo(z, by, tx)])\
               +confz[coo(z, ty, tx)] * (BXMzy * confy[coo(z, ty, bx)] + BYMzy * confy[coo(z, by, tx)])\
               +confz[coo(z, ty, tx)] * (BXMzz * confz[coo(z, ty, bx)] + BYMzz * confz[coo(z, by, tx)] + BZMzz * confz[coo(z-1, ty, tx)] - A * confz[coo(z, ty, tx)]);
    //Bottom-left corner
    sD[y][x] -= confx[coo(z, ty+1, tx)] * (BXMxx * confx[coo(z, ty+1, bx)] + BYMxx * confx[coo(z, ty, tx)] + BZMxx * confx[coo(z-1, ty+1, tx)])\
               +confx[coo(z, ty+1, tx)] * (BXMxy * confy[coo(z, ty+1, bx)] + BYMxy * confy[coo(z, ty, tx)] + BZMxy * confy[coo(z-1, ty+1, tx)])\
               +confx[coo(z, ty+1, tx)] * (BXMxz * confz[coo(z, ty+1, bx)] + BYMxz * confz[coo(z, ty, tx)])\
               +confy[coo(z, ty+1, tx)] * (BXMyx * confx[coo(z, ty+1, bx)] + BYMyx * confx[coo(z, ty, tx)] + BZMyx * confx[coo(z-1, ty+1, tx)])\
               +confy[coo(z, ty+1, tx)] * (BXMyy * confy[coo(z, ty+1, bx)] + BYMyy * confy[coo(z, ty, tx)] + BZMyy * confy[coo(z-1, ty+1, tx)])\
               +confy[coo(z, ty+1, tx)] * (BXMyz * confz[coo(z, ty+1, bx)] + BYMyz * confz[coo(z, ty, tx)])\
               +confz[coo(z, ty+1, tx)] * (BXMzx * confx[coo(z, ty+1, bx)] + BYMzx * confx[coo(z, ty, tx)])\
               +confz[coo(z, ty+1, tx)] * (BXMzy * confy[coo(z, ty+1, bx)] + BYMzy * confy[coo(z, ty, tx)])\
               +confz[coo(z, ty+1, tx)] * (BXMzz * confz[coo(z, ty+1, bx)] + BYMzz * confz[coo(z, ty, tx)] + BZMzz * confz[coo(z-1, ty+1, tx)] - A * confz[coo(z, ty+1, tx)]);
    //Top-right corner
    sD[y][x] -= confx[coo(z, ty, tx+1)] * (BXMxx * confx[coo(z, ty, tx)] + BYMxx * confx[coo(z, by, tx+1)] + BZMxx * confx[coo(z-1, ty, tx+1)])\
               +confx[coo(z, ty, tx+1)] * (BXMxy * confy[coo(z, ty, tx)] + BYMxy * confy[coo(z, by, tx+1)] + BZMxy * confy[coo(z-1, ty, tx+1)])\
               +confx[coo(z, ty, tx+1)] * (BXMxz * confz[coo(z, ty, tx)] + BYMxz * confz[coo(z, by, tx+1)])\
               +confy[coo(z, ty, tx+1)] * (BXMyx * confx[coo(z, ty, tx)] + BYMyx * confx[coo(z, by, tx+1)] + BZMyx * confx[coo(z-1, ty, tx+1)])\
               +confy[coo(z, ty, tx+1)] * (BXMyy * confy[coo(z, ty, tx)] + BYMyy * confy[coo(z, by, tx+1)] + BZMyy * confy[coo(z-1, ty, tx+1)])\
               +confy[coo(z, ty, tx+1)] * (BXMyz * confz[coo(z, ty, tx)] + BYMyz * confz[coo(z, by, tx+1)])\
               +confz[coo(z, ty, tx+1)] * (BXMzx * confx[coo(z, ty, tx)] + BYMzx * confx[coo(z, by, tx+1)])\
               +confz[coo(z, ty, tx+1)] * (BXMzy * confy[coo(z, ty, tx)] + BYMzy * confy[coo(z, by, tx+1)])\
               +confz[coo(z, ty, tx+1)] * (BXMzz * confz[coo(z, ty, tx)] + BYMzz * confz[coo(z, by, tx+1)] + BZMzz * confz[coo(z-1, ty, tx+1)] - A * confz[coo(z, ty, tx+1)]);
    //Bottom-right corner
    sD[y][x] -= confx[coo(z, ty+1, tx+1)] * (BXMxx * confx[coo(z, ty+1, tx)] + BYMxx * confx[coo(z, ty, tx+1)] + BZMxx * confx[coo(z-1, ty+1, tx+1)])\
               +confx[coo(z, ty+1, tx+1)] * (BXMxy * confy[coo(z, ty+1, tx)] + BYMxy * confy[coo(z, ty, tx+1)] + BZMxy * confy[coo(z-1, ty+1, tx+1)])\
               +confx[coo(z, ty+1, tx+1)] * (BXMxz * confz[coo(z, ty+1, tx)] + BYMxz * confz[coo(z, ty, tx+1)])\
               +confy[coo(z, ty+1, tx+1)] * (BXMyx * confx[coo(z, ty+1, tx)] + BYMyx * confx[coo(z, ty, tx+1)] + BZMyx * confx[coo(z-1, ty+1, tx+1)])\
               +confy[coo(z, ty+1, tx+1)] * (BXMyy * confy[coo(z, ty+1, tx)] + BYMyy * confy[coo(z, ty, tx+1)] + BZMyy * confy[coo(z-1, ty+1, tx+1)])\
               +confy[coo(z, ty+1, tx+1)] * (BXMyz * confz[coo(z, ty+1, tx)] + BYMyz * confz[coo(z, ty, tx+1)])\
               +confz[coo(z, ty+1, tx+1)] * (BXMzx * confx[coo(z, ty+1, tx)] + BYMzx * confx[coo(z, ty, tx+1)])\
               +confz[coo(z, ty+1, tx+1)] * (BXMzy * confy[coo(z, ty+1, tx)] + BYMzy * confy[coo(z, ty, tx+1)])\
               +confz[coo(z, ty+1, tx+1)] * (BXMzz * confz[coo(z, ty+1, tx)] + BYMzz * confz[coo(z, ty, tx+1)] + BZMzz * confz[coo(z-1, ty+1, tx+1)] - A * confz[coo(z, ty+1, tx+1)]);
  }
	__syncthreads();


	//Sum over all elements in each sD
	if(y < BlockSize_y/2)
		sD[y][x] += sD[y+BlockSize_y/2] [x];
	__syncthreads();
	if(y<BlockSize_y/4)
		sD[y][x] += sD[y+BlockSize_y/4] [x];
	__syncthreads();
	if(y<BlockSize_y/8)
		sD[y][x] += sD[y+BlockSize_y/8] [x];
	__syncthreads();
	if(y<BlockSize_y/16)
		sD[y][x] += sD[y+BlockSize_y/16] [x];
	__syncthreads();
	if(y==0 && x<BlockSize_x/2)
		sD[y][x] += sD[y][x+BlockSize_x/2] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/4)
		sD[y][x] += sD[y][x+BlockSize_x/4] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/8)
		sD[y][x] += sD[y][x+BlockSize_x/8] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/16)
		sD[y][x] += sD[y][x+BlockSize_x/16] ;
	__syncthreads();

	if(y==0 && x==0)
		out[dataoff + (blockIdx.x % BN)] = sD[0][0];
	__syncthreads();
	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
  sD[y][x]  = 0;
  for (z = 0; z < SpinSize_z; z++){
    sD[y][x] += confx[coo(z, ty, tx)];
    sD[y][x] += confx[coo(z, ty+1, tx)];
    sD[y][x] += confx[coo(z, ty ,tx+1)];
    sD[y][x] += confx[coo(z, ty+1, tx+1)];
  }
	__syncthreads();

	//Sum over all elements in each sD
	if(y < BlockSize_y/2)
		sD[y][x] += sD[y+BlockSize_y/2] [x];
	__syncthreads();
	if(y < BlockSize_y/4)
		sD[y][x] += sD[y+BlockSize_y/4] [x];
	__syncthreads();
	if(y < BlockSize_y/8)
		sD[y][x] += sD[y+BlockSize_y/8] [x];
	__syncthreads();
	if(y < BlockSize_y/16)
		sD[y][x] += sD[y+BlockSize_y/16] [x];
	__syncthreads();
	if(y==0 && x<BlockSize_x/2)
		sD[y][x] += sD[y][x+BlockSize_x/2] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/4)
		sD[y][x] += sD[y][x+BlockSize_x/4] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/8)
		sD[y][x] += sD[y][x+BlockSize_x/8] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/16)
		sD[y][x] += sD[y][x+BlockSize_x/16] ;
	__syncthreads();

	if(x==0 && y==0)
		out[dataoff + (blockIdx.x % BN) + BN] = sD[0][0];
	__syncthreads();

	//Sum over the magnetic moments in y direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	//Top-left corner
  sD[y][x]  = 0;
  for (z = 0; z < SpinSize_z; z++){
    sD[y][x] += confy[coo(z, ty, tx)];
    sD[y][x] += confy[coo(z, ty+1, tx)];
    sD[y][x] += confy[coo(z, ty ,tx+1)];
    sD[y][x] += confy[coo(z, ty+1, tx+1)];
  }
	__syncthreads();

	//Sum over all elements in each sD
	if(y < BlockSize_y/2)
		sD[y][x] += sD[y+BlockSize_y/2] [x];
	__syncthreads();
	if(y < BlockSize_y/4)
		sD[y][x] += sD[y+BlockSize_y/4] [x];
	__syncthreads();
	if(y < BlockSize_y/8)
		sD[y][x] += sD[y+BlockSize_y/8] [x];
	__syncthreads();
	if(y < BlockSize_y/16)
		sD[y][x] += sD[y+BlockSize_y/16] [x];
	__syncthreads();
	if(y==0 && x<BlockSize_x/2)
		sD[y][x] += sD[y][x+BlockSize_x/2] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/4)
		sD[y][x] += sD[y][x+BlockSize_x/4] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/8)
		sD[y][x] += sD[y][x+BlockSize_x/8] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/16)
		sD[y][x] += sD[y][x+BlockSize_x/16] ;
	__syncthreads();

	if(x==0 && y==0)
		out[dataoff + (blockIdx.x % BN) + 2 * BN] = sD[0][0];
	__syncthreads();

	//Sum over the magnetic moments in z direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
  sD[y][x]  = 0;
  for (z = 0; z < SpinSize_z; z++){
    sD[y][x] += confz[coo(z, ty, tx)];
    sD[y][x] += confz[coo(z, ty+1, tx)];
    sD[y][x] += confz[coo(z, ty ,tx+1)];
    sD[y][x] += confz[coo(z, ty+1, tx+1)];
  }
	__syncthreads();

	//Sum over all elements in each sD
	if(y < BlockSize_y/2)
		sD[y][x] += sD[y+BlockSize_y/2] [x];
	__syncthreads();
	if(y < BlockSize_y/4)
		sD[y][x] += sD[y+BlockSize_y/4] [x];
	__syncthreads();
	if(y < BlockSize_y/8)
		sD[y][x] += sD[y+BlockSize_y/8] [x];
	__syncthreads();
	if(y < BlockSize_y/16)
		sD[y][x] += sD[y+BlockSize_y/16] [x];
	__syncthreads();
	if(y==0 && x<BlockSize_x/2)
		sD[y][x] += sD[y][x+BlockSize_x/2] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/4)
		sD[y][x] += sD[y][x+BlockSize_x/4] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/8)
		sD[y][x] += sD[y][x+BlockSize_x/8] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/16)
		sD[y][x] += sD[y][x+BlockSize_x/16] ;
	__syncthreads();

	if(x==0 && y==0)
		out[dataoff + (blockIdx.x % BN) + 3 * BN] = sD[0][0];
	__syncthreads();
	//try to measure Chern number
	//Top-left corner
  sD[y][x]  = 0;
  for (z = 0; z < SpinSize_z; z++){
    sD[y][x]  = confx[coo(z, ty, tx)] * (
    (confy[coo(z, ty, tx)]-confy[coo(z, ty, bx)])*(confz[coo(z, ty, tx)]-confz[coo(z, by, tx)])
    -(confz[coo(z, ty, tx)]-confz[coo(z, ty, bx)])*(confy[coo(z, ty, tx)]-confy[coo(z, by, tx)])
    )+confy[coo(z, ty, tx)] * (
    (confz[coo(z, ty, tx)]-confz[coo(z, ty, bx)])*(confx[coo(z, ty, tx)]-confx[coo(z, by, tx)])
    -(confx[coo(z, ty, tx)]-confx[coo(z, ty, bx)])*(confz[coo(z, ty, tx)]-confz[coo(z, by, tx)])
    )+confz[coo(z, ty, tx)] * (
    (confx[coo(z, ty, tx)]-confx[coo(z, ty, bx)])*(confy[coo(z, ty, tx)]-confy[coo(z, by, tx)])
    -(confy[coo(z, ty, tx)]-confy[coo(z, ty, bx)])*(confx[coo(z, ty, tx)]-confx[coo(z, by, tx)])
    );
    //Bottom-left corner
    sD[y][x] += confx[coo(z, typ, tx)] * (
    (confy[coo(z, typ, tx)]-confy[coo(z, typ, bx)])*(confz[coo(z, typ, tx)]-confz[coo(z, ty, tx)])
    -(confz[coo(z, typ, tx)]-confz[coo(z, typ, bx)])*(confy[coo(z, typ, tx)]-confy[coo(z, ty, tx)])
    )+confy[coo(z, typ, tx)]*(
    (confz[coo(z, typ, tx)]-confz[coo(z, typ, bx)])*(confx[coo(z, typ, tx)]-confx[coo(z, ty, tx)])
    -(confx[coo(z, typ, tx)]-confx[coo(z, typ, bx)])*(confz[coo(z, typ, tx)]-confz[coo(z, ty, tx)])
    )+confz[coo(z, typ, tx)] * (
    (confx[coo(z, typ, tx)]-confx[coo(z, typ, bx)])*(confy[coo(z, typ, tx)]-confy[coo(z, ty, tx)])
    -(confy[coo(z, typ, tx)]-confy[coo(z, typ, bx)])*(confx[coo(z, typ, tx)]-confx[coo(z, ty, tx)])
    );
    //Top-right corner
    sD[y][x] += confx[coo(z, ty, txp)] * (
    (confy[coo(z, ty, txp)]-confy[coo(z, ty, tx)])*(confz[coo(z, ty, txp)]-confz[coo(z, by, txp)])
    -(confz[coo(z, ty, txp)]-confz[coo(z, ty, tx)])*(confy[coo(z, ty, txp)]-confy[coo(z, by, txp)])
    )+confy[coo(z, ty, txp)]*(
    (confz[coo(z, ty, txp)]-confz[coo(z, ty, tx)])*(confx[coo(z, ty, txp)]-confx[coo(z, by, txp)])
    -(confx[coo(z, ty, txp)]-confx[coo(z, ty, tx)])*(confz[coo(z, ty, txp)]-confz[coo(z, by, txp)])
    )+confz[coo(z, ty, txp)] * (
    (confx[coo(z, ty, txp)]-confx[coo(z, ty, tx)])*(confy[coo(z, ty, txp)]-confy[coo(z, by, txp)])
    -(confy[coo(z, ty, txp)]-confy[coo(z, ty, tx)])*(confx[coo(z, ty, txp)]-confx[coo(z, by, txp)])
    );
    //Bottom-right corner
    sD[y][x] += confx[coo(z, typ, txp)] * (
    (confy[coo(z, typ, txp)]-confy[coo(z, typ, tx)])*(confz[coo(z, typ, txp)]-confz[coo(z, ty, txp)])
    -(confz[coo(z, typ, txp)]-confz[coo(z, typ, tx)])*(confy[coo(z, typ, txp)]-confy[coo(z, ty, txp)])
    )+confy[coo(z, typ, txp)]*(
    (confz[coo(z, typ, txp)]-confz[coo(z, typ, tx)])*(confx[coo(z, typ, txp)]-confx[coo(z, ty, txp)])
    -(confx[coo(z, typ, txp)]-confx[coo(z, typ, tx)])*(confz[coo(z, typ, txp)]-confz[coo(z, ty, txp)])
    )+confz[coo(z, typ, txp)] * (
    (confx[coo(z, typ, txp)]-confx[coo(z, typ, tx)])*(confy[coo(z, typ, txp)]-confy[coo(z, ty, txp)])
    -(confy[coo(z, typ, txp)]-confy[coo(z, typ, tx)])*(confx[coo(z, typ, txp)]-confx[coo(z, ty, txp)])
    );
  }
	__syncthreads();

	//Sum over all elements in each sD
	if(y < BlockSize_y/2)
		sD[y][x] += sD[y+BlockSize_y/2] [x];
	__syncthreads();
	if(y < BlockSize_y/4)
		sD[y][x] += sD[y+BlockSize_y/4] [x];
	__syncthreads();
	if(y < BlockSize_y/8)
		sD[y][x] += sD[y+BlockSize_y/8] [x];
	__syncthreads();
	if(y < BlockSize_y/16)
		sD[y][x] += sD[y+BlockSize_y/16] [x];
	__syncthreads();
	if(y==0 && x<BlockSize_x/2)
		sD[y][x] += sD[y][x+BlockSize_x/2] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/4)
		sD[y][x] += sD[y][x+BlockSize_x/4] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/8)
		sD[y][x] += sD[y][x+BlockSize_x/8] ;
	__syncthreads();
	if(y==0 && x<BlockSize_x/16)
		sD[y][x] += sD[y][x+BlockSize_x/16] ;
	__syncthreads();

	if(x==0 && y==0)
		out[dataoff + (blockIdx.x % BN) + 4 * BN] = sD[0][0];
	__syncthreads();
}

#endif
