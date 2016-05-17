#include "measurements.cuh"
#ifdef TRI
__global__ void calTRI(float *confx, float *confy, float *confz, double *out){
	//Energy variables
	__shared__ double sD[BlockSize_y][BlockSize_x];
	const int x = threadIdx.x % (BlockSize_x);
	const int y = (threadIdx.x / BlockSize_x);
	const int tx = 3 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
	const int ty =(blockIdx.x / BN) * SpinSize +  3 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
	const int txp = tx +1 ;
	const int typ = ty +1 ;
	const int txp2 = tx +2 ;
	const int typ2 = ty +2 ;
	//const int ty = 2 * ((blockIdx.x / BN) * SpinSize + ((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
	const int dataoff = (blockIdx.x / BN) * MEASURE_NUM * BN;
	int bx, by;
	//-----Calculate the energy of each spin pairs in the system-----
	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.

	bx = (tx + SpinSize - 1) % SpinSize;
	if((ty % SpinSize) == 0)	by = ty + SpinSize - 1;
	else				by = ty - 1;
	//Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.

	//0,0
	sD[y][x] = -confx[coo2D(ty, tx)] * ( BXMxx * confx[coo2D(ty, bx)] + BYMxx * confx[coo2D(by, tx)] + BWMxx * confx[coo2D(by, bx)])\
	           -confx[coo2D(ty, tx)] * ( BXMxy * confy[coo2D(ty, bx)] + BYMxy * confy[coo2D(by, tx)] + BWMxy * confy[coo2D(by, bx)])\
	           -confx[coo2D(ty, tx)] * ( BXMxz * confz[coo2D(ty, bx)] + BYMxz * confz[coo2D(by, tx)] + BWMxz * confz[coo2D(by, bx)])\
		         -confy[coo2D(ty, tx)] * ( BXMyx * confx[coo2D(ty, bx)] + BYMyx * confx[coo2D(by, tx)] + BWMyx * confx[coo2D(by, bx)])\
		         -confy[coo2D(ty, tx)] * ( BXMyy * confy[coo2D(ty, bx)] + BYMyy * confy[coo2D(by, tx)] + BWMyy * confy[coo2D(by, bx)])\
		         -confy[coo2D(ty, tx)] * ( BXMyz * confz[coo2D(ty, bx)] + BYMyz * confz[coo2D(by, tx)] + BWMyz * confz[coo2D(by, bx)])\
		         -confz[coo2D(ty, tx)] * ( BXMzx * confx[coo2D(ty, bx)] + BYMzx * confx[coo2D(by, tx)] + BWMzx * confx[coo2D(by, bx)])\
		         -confz[coo2D(ty, tx)] * ( BXMzy * confy[coo2D(ty, bx)] + BYMzy * confy[coo2D(by, tx)] + BWMzy * confy[coo2D(by, bx)])\
		         -confz[coo2D(ty, tx)] * ( BXMzz * confz[coo2D(ty, bx)] + BYMzz * confz[coo2D(by, tx)] + BWMzz * confz[coo2D(by, bx)] - A * confz[coo2D(ty, tx)]);
	//1,0
	sD[y][x] -= confx[coo2D(typ, tx)] * ( BXMxx * confx[coo2D(typ, bx)] + BYMxx * confx[coo2D(ty, tx)] + BWMxx * confx[coo2D(ty, bx)])\
		         +confx[coo2D(typ, tx)] * ( BXMxy * confy[coo2D(typ, bx)] + BYMxy * confy[coo2D(ty, tx)] + BWMxy * confy[coo2D(ty, bx)])\
		         +confx[coo2D(typ, tx)] * ( BXMxz * confz[coo2D(typ, bx)] + BYMxz * confz[coo2D(ty, tx)] + BWMxz * confz[coo2D(ty, bx)])\
		         +confy[coo2D(typ, tx)] * ( BXMyx * confx[coo2D(typ, bx)] + BYMyx * confx[coo2D(ty, tx)] + BWMyx * confx[coo2D(ty, bx)])\
		         +confy[coo2D(typ, tx)] * ( BXMyy * confy[coo2D(typ, bx)] + BYMyy * confy[coo2D(ty, tx)] + BWMyy * confy[coo2D(ty, bx)])\
		         +confy[coo2D(typ, tx)] * ( BXMyz * confz[coo2D(typ, bx)] + BYMyz * confz[coo2D(ty, tx)] + BWMyz * confz[coo2D(ty, bx)])\
		         +confz[coo2D(typ, tx)] * ( BXMzx * confx[coo2D(typ, bx)] + BYMzx * confx[coo2D(ty, tx)] + BWMzx * confx[coo2D(ty, bx)])\
		         +confz[coo2D(typ, tx)] * ( BXMzy * confy[coo2D(typ, bx)] + BYMzy * confy[coo2D(ty, tx)] + BWMzy * confy[coo2D(ty, bx)])\
		         +confz[coo2D(typ, tx)] * ( BXMzz * confz[coo2D(typ, bx)] + BYMzz * confz[coo2D(ty, tx)] + BWMzz * confz[coo2D(ty, bx)] - A * confz[coo2D((ty+1), tx)]);
	//2,0
	sD[y][x] -= confx[coo2D(typ2, tx)] * ( BXMxx * confx[coo2D(typ2, bx)] + BYMxx * confx[coo2D(typ, tx)] + BWMxx * confx[coo2D(typ, bx)])\
		         +confx[coo2D(typ2, tx)] * ( BXMxy * confy[coo2D(typ2, bx)] + BYMxy * confy[coo2D(typ, tx)] + BWMxy * confy[coo2D(typ, bx)])\
		         +confx[coo2D(typ2, tx)] * ( BXMxz * confz[coo2D(typ2, bx)] + BYMxz * confz[coo2D(typ, tx)] + BWMxz * confz[coo2D(typ, bx)])\
		         +confy[coo2D(typ2, tx)] * ( BXMyx * confx[coo2D(typ2, bx)] + BYMyx * confx[coo2D(typ, tx)] + BWMyx * confx[coo2D(typ, bx)])\
		         +confy[coo2D(typ2, tx)] * ( BXMyy * confy[coo2D(typ2, bx)] + BYMyy * confy[coo2D(typ, tx)] + BWMyy * confy[coo2D(typ, bx)])\
		         +confy[coo2D(typ2, tx)] * ( BXMyz * confz[coo2D(typ2, bx)] + BYMyz * confz[coo2D(typ, tx)] + BWMyz * confz[coo2D(typ, bx)])\
		         +confz[coo2D(typ2, tx)] * ( BXMzx * confx[coo2D(typ2, bx)] + BYMzx * confx[coo2D(typ, tx)] + BWMzx * confx[coo2D(typ, bx)])\
		         +confz[coo2D(typ2, tx)] * ( BXMzy * confy[coo2D(typ2, bx)] + BYMzy * confy[coo2D(typ, tx)] + BWMzy * confy[coo2D(typ, bx)])\
		         +confz[coo2D(typ2, tx)] * ( BXMzz * confz[coo2D(typ2, bx)] + BYMzz * confz[coo2D(typ, tx)] + BWMzz * confz[coo2D(typ, bx)] - A * confz[coo2D((ty+1), tx)]);
	//0,1
	sD[y][x] -= confx[coo2D(ty, txp)] * ( BXMxx * confx[coo2D(ty, tx)] + BYMxx * confx[coo2D(by, txp)] + BWMxx * confx[coo2D(by, tx)])\
		         +confx[coo2D(ty, txp)] * ( BXMxy * confy[coo2D(ty, tx)] + BYMxy * confy[coo2D(by, txp)] + BWMxy * confy[coo2D(by, tx)])\
		         +confx[coo2D(ty, txp)] * ( BXMxz * confz[coo2D(ty, tx)] + BYMxz * confz[coo2D(by, txp)] + BWMxz * confz[coo2D(by, tx)])\
		         +confy[coo2D(ty, txp)] * ( BXMyx * confx[coo2D(ty, tx)] + BYMyx * confx[coo2D(by, txp)] + BWMyx * confx[coo2D(by, tx)])\
		         +confy[coo2D(ty, txp)] * ( BXMyy * confy[coo2D(ty, tx)] + BYMyy * confy[coo2D(by, txp)] + BWMyy * confy[coo2D(by, tx)])\
		         +confy[coo2D(ty, txp)] * ( BXMyz * confz[coo2D(ty, tx)] + BYMyz * confz[coo2D(by, txp)] + BWMyz * confz[coo2D(by, tx)])\
		         +confz[coo2D(ty, txp)] * ( BXMzx * confx[coo2D(ty, tx)] + BYMzx * confx[coo2D(by, txp)] + BWMzx * confx[coo2D(by, tx)])\
		         +confz[coo2D(ty, txp)] * ( BXMzy * confy[coo2D(ty, tx)] + BYMzy * confy[coo2D(by, txp)] + BWMzy * confy[coo2D(by, tx)])\
		         +confz[coo2D(ty, txp)] * ( BXMzz * confz[coo2D(ty, tx)] + BYMzz * confz[coo2D(by, txp)] + BWMzz * confz[coo2D(by, tx)] - A * confz[coo2D(ty, tx+1)]);
	//1,1
	sD[y][x] -= confx[coo2D(typ, txp)] * ( BXMxx * confx[coo2D(typ, tx)] + BYMxx * confx[coo2D(ty, txp)] + BWMxx * confx[coo2D(ty, tx)])\
		         +confx[coo2D(typ, txp)] * ( BXMxy * confy[coo2D(typ, tx)] + BYMxy * confy[coo2D(ty, txp)] + BWMxy * confy[coo2D(ty, tx)])\
		         +confx[coo2D(typ, txp)] * ( BXMxz * confz[coo2D(typ, tx)] + BYMxz * confz[coo2D(ty, txp)] + BWMxz * confz[coo2D(ty, tx)])\
		         +confy[coo2D(typ, txp)] * ( BXMyx * confx[coo2D(typ, tx)] + BYMyx * confx[coo2D(ty, txp)] + BWMyx * confx[coo2D(ty, tx)])\
		         +confy[coo2D(typ, txp)] * ( BXMyy * confy[coo2D(typ, tx)] + BYMyy * confy[coo2D(ty, txp)] + BWMyy * confy[coo2D(ty, tx)])\
		         +confy[coo2D(typ, txp)] * ( BXMyz * confz[coo2D(typ, tx)] + BYMyz * confz[coo2D(ty, txp)] + BWMyz * confz[coo2D(ty, tx)])\
		         +confz[coo2D(typ, txp)] * ( BXMzx * confx[coo2D(typ, tx)] + BYMzx * confx[coo2D(ty, txp)] + BWMzx * confx[coo2D(ty, tx)])\
		         +confz[coo2D(typ, txp)] * ( BXMzy * confy[coo2D(typ, tx)] + BYMzy * confy[coo2D(ty, txp)] + BWMzy * confy[coo2D(ty, tx)])\
		         +confz[coo2D(typ, txp)] * ( BXMzz * confz[coo2D(typ, tx)] + BYMzz * confz[coo2D(ty, txp)] + BWMzz * confz[coo2D(ty, tx)] - A * confz[coo2D(ty, tx+1)]);
	//2,1
	sD[y][x] -= confx[coo2D(typ2, txp)] * ( BXMxx * confx[coo2D(typ2, tx)] + BYMxx * confx[coo2D(typ, txp)] + BWMxx * confx[coo2D(typ, tx)])\
		         +confx[coo2D(typ2, txp)] * ( BXMxy * confy[coo2D(typ2, tx)] + BYMxy * confy[coo2D(typ, txp)] + BWMxy * confy[coo2D(typ, tx)])\
		         +confx[coo2D(typ2, txp)] * ( BXMxz * confz[coo2D(typ2, tx)] + BYMxz * confz[coo2D(typ, txp)] + BWMxz * confz[coo2D(typ, tx)])\
		         +confy[coo2D(typ2, txp)] * ( BXMyx * confx[coo2D(typ2, tx)] + BYMyx * confx[coo2D(typ, txp)] + BWMyx * confx[coo2D(typ, tx)])\
		         +confy[coo2D(typ2, txp)] * ( BXMyy * confy[coo2D(typ2, tx)] + BYMyy * confy[coo2D(typ, txp)] + BWMyy * confy[coo2D(typ, tx)])\
		         +confy[coo2D(typ2, txp)] * ( BXMyz * confz[coo2D(typ2, tx)] + BYMyz * confz[coo2D(typ, txp)] + BWMyz * confz[coo2D(typ, tx)])\
		         +confz[coo2D(typ2, txp)] * ( BXMzx * confx[coo2D(typ2, tx)] + BYMzx * confx[coo2D(typ, txp)] + BWMzx * confx[coo2D(typ, tx)])\
		         +confz[coo2D(typ2, txp)] * ( BXMzy * confy[coo2D(typ2, tx)] + BYMzy * confy[coo2D(typ, txp)] + BWMzy * confy[coo2D(typ, tx)])\
		         +confz[coo2D(typ2, txp)] * ( BXMzz * confz[coo2D(typ2, tx)] + BYMzz * confz[coo2D(typ, txp)] + BWMzz * confz[coo2D(typ, tx)] - A * confz[coo2D(ty, tx+1)]);
	//0,2
	sD[y][x] -= confx[coo2D(ty, txp2)] * ( BXMxx * confx[coo2D(ty, txp)] + BYMxx * confx[coo2D(by, txp2)] + BWMxx * confx[coo2D(by, txp)])\
		         +confx[coo2D(ty, txp2)] * ( BXMxy * confy[coo2D(ty, txp)] + BYMxy * confy[coo2D(by, txp2)] + BWMxy * confy[coo2D(by, txp)])\
		         +confx[coo2D(ty, txp2)] * ( BXMxz * confz[coo2D(ty, txp)] + BYMxz * confz[coo2D(by, txp2)] + BWMxz * confz[coo2D(by, txp)])\
		         +confy[coo2D(ty, txp2)] * ( BXMyx * confx[coo2D(ty, txp)] + BYMyx * confx[coo2D(by, txp2)] + BWMyx * confx[coo2D(by, txp)])\
		         +confy[coo2D(ty, txp2)] * ( BXMyy * confy[coo2D(ty, txp)] + BYMyy * confy[coo2D(by, txp2)] + BWMyy * confy[coo2D(by, txp)])\
		         +confy[coo2D(ty, txp2)] * ( BXMyz * confz[coo2D(ty, txp)] + BYMyz * confz[coo2D(by, txp2)] + BWMyz * confz[coo2D(by, txp)])\
		         +confz[coo2D(ty, txp2)] * ( BXMzx * confx[coo2D(ty, txp)] + BYMzx * confx[coo2D(by, txp2)] + BWMzx * confx[coo2D(by, txp)])\
		         +confz[coo2D(ty, txp2)] * ( BXMzy * confy[coo2D(ty, txp)] + BYMzy * confy[coo2D(by, txp2)] + BWMzy * confy[coo2D(by, txp)])\
		         +confz[coo2D(ty, txp2)] * ( BXMzz * confz[coo2D(ty, txp)] + BYMzz * confz[coo2D(by, txp2)] + BWMzz * confz[coo2D(by, txp)] - A * confz[coo2D(ty, tx+1)]);
	//1,2
	sD[y][x] -= confx[coo2D(typ, txp2)] * ( BXMxx * confx[coo2D(typ, txp)] + BYMxx * confx[coo2D(ty, txp2)] + BWMxx * confx[coo2D(ty, txp)])\
		         +confx[coo2D(typ, txp2)] * ( BXMxy * confy[coo2D(typ, txp)] + BYMxy * confy[coo2D(ty, txp2)] + BWMxy * confy[coo2D(ty, txp)])\
		         +confx[coo2D(typ, txp2)] * ( BXMxz * confz[coo2D(typ, txp)] + BYMxz * confz[coo2D(ty, txp2)] + BWMxz * confz[coo2D(ty, txp)])\
		         +confy[coo2D(typ, txp2)] * ( BXMyx * confx[coo2D(typ, txp)] + BYMyx * confx[coo2D(ty, txp2)] + BWMyx * confx[coo2D(ty, txp)])\
		         +confy[coo2D(typ, txp2)] * ( BXMyy * confy[coo2D(typ, txp)] + BYMyy * confy[coo2D(ty, txp2)] + BWMyy * confy[coo2D(ty, txp)])\
		         +confy[coo2D(typ, txp2)] * ( BXMyz * confz[coo2D(typ, txp)] + BYMyz * confz[coo2D(ty, txp2)] + BWMyz * confz[coo2D(ty, txp)])\
		         +confz[coo2D(typ, txp2)] * ( BXMzx * confx[coo2D(typ, txp)] + BYMzx * confx[coo2D(ty, txp2)] + BWMzx * confx[coo2D(ty, txp)])\
		         +confz[coo2D(typ, txp2)] * ( BXMzy * confy[coo2D(typ, txp)] + BYMzy * confy[coo2D(ty, txp2)] + BWMzy * confy[coo2D(ty, txp)])\
		         +confz[coo2D(typ, txp2)] * ( BXMzz * confz[coo2D(typ, txp)] + BYMzz * confz[coo2D(ty, txp2)] + BWMzz * confz[coo2D(ty, txp)] - A * confz[coo2D(ty, tx+1)]);
	//2,2
	sD[y][x] -= confx[coo2D(typ2, txp2)] * ( BXMxx * confx[coo2D(typ2, txp)] + BYMxx * confx[coo2D(typ, txp2)] + BWMxx * confx[coo2D(typ, txp)])\
		         +confx[coo2D(typ2, txp2)] * ( BXMxy * confy[coo2D(typ2, txp)] + BYMxy * confy[coo2D(typ, txp2)] + BWMxy * confy[coo2D(typ, txp)])\
		         +confx[coo2D(typ2, txp2)] * ( BXMxz * confz[coo2D(typ2, txp)] + BYMxz * confz[coo2D(typ, txp2)] + BWMxz * confz[coo2D(typ, txp)])\
		         +confy[coo2D(typ2, txp2)] * ( BXMyx * confx[coo2D(typ2, txp)] + BYMyx * confx[coo2D(typ, txp2)] + BWMyx * confx[coo2D(typ, txp)])\
		         +confy[coo2D(typ2, txp2)] * ( BXMyy * confy[coo2D(typ2, txp)] + BYMyy * confy[coo2D(typ, txp2)] + BWMyy * confy[coo2D(typ, txp)])\
		         +confy[coo2D(typ2, txp2)] * ( BXMyz * confz[coo2D(typ2, txp)] + BYMyz * confz[coo2D(typ, txp2)] + BWMyz * confz[coo2D(typ, txp)])\
		         +confz[coo2D(typ2, txp2)] * ( BXMzx * confx[coo2D(typ2, txp)] + BYMzx * confx[coo2D(typ, txp2)] + BWMzx * confx[coo2D(typ, txp)])\
		         +confz[coo2D(typ2, txp2)] * ( BXMzy * confy[coo2D(typ2, txp)] + BYMzy * confy[coo2D(typ, txp2)] + BWMzy * confy[coo2D(typ, txp)])\
		         +confz[coo2D(typ2, txp2)] * ( BXMzz * confz[coo2D(typ2, txp)] + BYMzz * confz[coo2D(typ, txp2)] + BWMzz * confz[coo2D(typ, txp)] - A * confz[coo2D(ty, tx+1)]);
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
	sD[y][x]  = confx[coo2D(ty, tx)];
	sD[y][x] += confx[coo2D(typ, tx)];
	sD[y][x] += confx[coo2D(typ2, tx)];
	sD[y][x] += confx[coo2D(ty, txp)];
	sD[y][x] += confx[coo2D(typ, txp)];
	sD[y][x] += confx[coo2D(typ2, txp)];
	sD[y][x] += confx[coo2D(ty, txp2)];
	sD[y][x] += confx[coo2D(typ, txp2)];
	sD[y][x] += confx[coo2D(typ2, txp2)];
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
	sD[y][x]  = confy[coo2D(ty, tx)];
	sD[y][x] += confy[coo2D(typ, tx)];
	sD[y][x] += confy[coo2D(typ2, tx)];
	sD[y][x] += confy[coo2D(ty, txp)];
	sD[y][x] += confy[coo2D(typ, txp)];
	sD[y][x] += confy[coo2D(typ2, txp)];
	sD[y][x] += confy[coo2D(ty, txp2)];
	sD[y][x] += confy[coo2D(typ, txp2)];
	sD[y][x] += confy[coo2D(typ2, txp2)];
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
	sD[y][x]  = confy[coo2D(ty, tx)];
	sD[y][x] += confy[coo2D(typ, tx)];
	sD[y][x] += confy[coo2D(typ2, tx)];
	sD[y][x] += confy[coo2D(ty, txp)];
	sD[y][x] += confy[coo2D(typ, txp)];
	sD[y][x] += confy[coo2D(typ2, txp)];
	sD[y][x] += confy[coo2D(ty, txp2)];
	sD[y][x] += confy[coo2D(typ, txp2)];
	sD[y][x] += confy[coo2D(typ2, txp2)];
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
	//(0,0)
	sD[y][x]  = confx[coo2D(ty, tx)] * (
	 (confy[coo2D(ty, tx)]-confy[coo2D(ty, bx)])*(2*confz[coo2D(ty, tx)]-confz[coo2D(by, tx)]-confz[coo2D(by, bx)])
	-(confz[coo2D(ty, tx)]-confz[coo2D(ty, bx)])*(2*confy[coo2D(ty, tx)]-confy[coo2D(by, tx)]-confy[coo2D(by, bx)])
	)+confy[coo2D(ty, tx)] * (
	 (confz[coo2D(ty, tx)]-confz[coo2D(ty, bx)])*(2*confx[coo2D(ty, tx)]-confx[coo2D(by, tx)]-confx[coo2D(by, bx)])
	-(confx[coo2D(ty, tx)]-confx[coo2D(ty, bx)])*(2*confz[coo2D(ty, tx)]-confz[coo2D(by, tx)]-confz[coo2D(by, bx)])
	)+confz[coo2D(ty, tx)] * (
	 (confx[coo2D(ty, tx)]-confx[coo2D(ty, bx)])*(2*confy[coo2D(ty, tx)]-confy[coo2D(by, tx)]-confy[coo2D(by, bx)])
	-(confy[coo2D(ty, tx)]-confy[coo2D(ty, bx)])*(2*confx[coo2D(ty, tx)]-confx[coo2D(by, tx)]-confx[coo2D(by, bx)])
	);
	//(1,0)
	sD[y][x] += confx[coo2D(typ, tx)] * (
	 (confy[coo2D(typ, tx)]-confy[coo2D(typ, bx)])*(2*confz[coo2D(typ, tx)]-confz[coo2D(ty, tx)]-confz[coo2D(ty, bx)])
	-(confz[coo2D(typ, tx)]-confz[coo2D(typ, bx)])*(2*confy[coo2D(typ, tx)]-confy[coo2D(ty, tx)]-confy[coo2D(ty, bx)])
	)+confy[coo2D(typ, tx)]*(
	 (confz[coo2D(typ, tx)]-confz[coo2D(typ, bx)])*(2*confx[coo2D(typ, tx)]-confx[coo2D(ty, tx)]-confx[coo2D(ty, bx)])
	-(confx[coo2D(typ, tx)]-confx[coo2D(typ, bx)])*(2*confz[coo2D(typ, tx)]-confz[coo2D(ty, tx)]-confz[coo2D(ty, bx)])
	)+confz[coo2D(typ, tx)] * (
	 (confx[coo2D(typ, tx)]-confx[coo2D(typ, bx)])*(2*confy[coo2D(typ, tx)]-confy[coo2D(ty, tx)]-confy[coo2D(ty, bx)])
	-(confy[coo2D(typ, tx)]-confy[coo2D(typ, bx)])*(2*confx[coo2D(typ, tx)]-confx[coo2D(ty, tx)]-confx[coo2D(ty, bx)])
	);
	//(2,0)
	sD[y][x] += confx[coo2D(typ2, tx)] * (
	 (confy[coo2D(typ2, tx)]-confy[coo2D(typ2, bx)])*(2*confz[coo2D(typ2, tx)]-confz[coo2D(typ, tx)]-confz[coo2D(typ, bx)])
	-(confz[coo2D(typ2, tx)]-confz[coo2D(typ2, bx)])*(2*confy[coo2D(typ2, tx)]-confy[coo2D(typ, tx)]-confy[coo2D(typ, bx)])
	)+confy[coo2D(typ2, tx)]*(
	 (confz[coo2D(typ2, tx)]-confz[coo2D(typ2, bx)])*(2*confx[coo2D(typ2, tx)]-confx[coo2D(typ, tx)]-confx[coo2D(typ, bx)])
	-(confx[coo2D(typ2, tx)]-confx[coo2D(typ2, bx)])*(2*confz[coo2D(typ2, tx)]-confz[coo2D(typ, tx)]-confz[coo2D(typ, bx)])
	)+confz[coo2D(typ2, tx)] * (
	 (confx[coo2D(typ2, tx)]-confx[coo2D(typ2, bx)])*(2*confy[coo2D(typ2, tx)]-confy[coo2D(typ, tx)]-confy[coo2D(typ, bx)])
	-(confy[coo2D(typ2, tx)]-confy[coo2D(typ2, bx)])*(2*confx[coo2D(typ2, tx)]-confx[coo2D(typ, tx)]-confx[coo2D(typ, bx)])
	);
	//(0,1)
	sD[y][x] += confx[coo2D(ty, txp)] * (
	 (confy[coo2D(ty, txp)]-confy[coo2D(ty, tx)])*(2*confz[coo2D(ty, txp)]-confz[coo2D(by, txp)]-confz[coo2D(by, tx)])
	-(confz[coo2D(ty, txp)]-confz[coo2D(ty, tx)])*(2*confy[coo2D(ty, txp)]-confy[coo2D(by, txp)]-confy[coo2D(by, tx)])
	)+confy[coo2D(ty, txp)]*(                                                                                        
	 (confz[coo2D(ty, txp)]-confz[coo2D(ty, tx)])*(2*confx[coo2D(ty, txp)]-confx[coo2D(by, txp)]-confx[coo2D(by, tx)])
	-(confx[coo2D(ty, txp)]-confx[coo2D(ty, tx)])*(2*confz[coo2D(ty, txp)]-confz[coo2D(by, txp)]-confz[coo2D(by, tx)])
	)+confz[coo2D(ty, txp)] * (                                                                                      
	 (confx[coo2D(ty, txp)]-confx[coo2D(ty, tx)])*(2*confy[coo2D(ty, txp)]-confy[coo2D(by, txp)]-confy[coo2D(by, tx)])
	-(confy[coo2D(ty, txp)]-confy[coo2D(ty, tx)])*(2*confx[coo2D(ty, txp)]-confx[coo2D(by, txp)]-confx[coo2D(by, tx)])
	);
	//(1,1)
	sD[y][x] += confx[coo2D(typ, txp)] * (
	 (confy[coo2D(typ, txp)]-confy[coo2D(typ, tx)])*(2*confz[coo2D(typ, txp)]-confz[coo2D(ty, txp)]-confz[coo2D(ty, tx)])
	-(confz[coo2D(typ, txp)]-confz[coo2D(typ, tx)])*(2*confy[coo2D(typ, txp)]-confy[coo2D(ty, txp)]-confy[coo2D(ty, tx)])
	)+confy[coo2D(typ, txp)]*(
	 (confz[coo2D(typ, txp)]-confz[coo2D(typ, tx)])*(2*confx[coo2D(typ, txp)]-confx[coo2D(ty, txp)]-confx[coo2D(ty, tx)])
	-(confx[coo2D(typ, txp)]-confx[coo2D(typ, tx)])*(2*confz[coo2D(typ, txp)]-confz[coo2D(ty, txp)]-confz[coo2D(ty, tx)])
	)+confz[coo2D(typ, txp)] * (
	 (confx[coo2D(typ, txp)]-confx[coo2D(typ, tx)])*(2*confy[coo2D(typ, txp)]-confy[coo2D(ty, txp)]-confy[coo2D(ty, tx)])
	-(confy[coo2D(typ, txp)]-confy[coo2D(typ, tx)])*(2*confx[coo2D(typ, txp)]-confx[coo2D(ty, txp)]-confx[coo2D(ty, tx)])
	);
	//(2,1)
	sD[y][x] += confx[coo2D(typ2, txp)] * (
	 (confy[coo2D(typ2, txp)]-confy[coo2D(typ2, tx)])*(2*confz[coo2D(typ2, txp)]-confz[coo2D(typ, txp)]-confz[coo2D(typ, tx)])
	-(confz[coo2D(typ2, txp)]-confz[coo2D(typ2, tx)])*(2*confy[coo2D(typ2, txp)]-confy[coo2D(typ, txp)]-confy[coo2D(typ, tx)])
	)+confy[coo2D(typ2, txp)]*(
	 (confz[coo2D(typ2, txp)]-confz[coo2D(typ2, tx)])*(2*confx[coo2D(typ2, txp)]-confx[coo2D(typ, txp)]-confx[coo2D(typ, tx)])
	-(confx[coo2D(typ2, txp)]-confx[coo2D(typ2, tx)])*(2*confz[coo2D(typ2, txp)]-confz[coo2D(typ, txp)]-confz[coo2D(typ, tx)])
	)+confz[coo2D(typ2, txp)] * (
	 (confx[coo2D(typ2, txp)]-confx[coo2D(typ2, tx)])*(2*confy[coo2D(typ2, txp)]-confy[coo2D(typ, txp)]-confy[coo2D(typ, tx)])
	-(confy[coo2D(typ2, txp)]-confy[coo2D(typ2, tx)])*(2*confx[coo2D(typ2, txp)]-confx[coo2D(typ, txp)]-confx[coo2D(typ, tx)])
	);
	//(0,2)
	sD[y][x] += confx[coo2D(ty, txp2)] * (
	 (confy[coo2D(ty, txp2)]-confy[coo2D(ty, txp)])*(2*confz[coo2D(ty, txp2)]-confz[coo2D(by, txp2)]-confz[coo2D(by, txp)])
	-(confz[coo2D(ty, txp2)]-confz[coo2D(ty, txp)])*(2*confy[coo2D(ty, txp2)]-confy[coo2D(by, txp2)]-confy[coo2D(by, txp)])
	)+confy[coo2D(ty, txp2)]*(                                                                                          
	 (confz[coo2D(ty, txp2)]-confz[coo2D(ty, txp)])*(2*confx[coo2D(ty, txp2)]-confx[coo2D(by, txp2)]-confx[coo2D(by, txp)])
	-(confx[coo2D(ty, txp2)]-confx[coo2D(ty, txp)])*(2*confz[coo2D(ty, txp2)]-confz[coo2D(by, txp2)]-confz[coo2D(by, txp)])
	)+confz[coo2D(ty, txp2)] * (                                                                                        
	 (confx[coo2D(ty, txp2)]-confx[coo2D(ty, txp)])*(2*confy[coo2D(ty, txp2)]-confy[coo2D(by, txp2)]-confy[coo2D(by, txp)])
	-(confy[coo2D(ty, txp2)]-confy[coo2D(ty, txp)])*(2*confx[coo2D(ty, txp2)]-confx[coo2D(by, txp2)]-confx[coo2D(by, txp)])
	);
	//(1,2)
	sD[y][x] += confx[coo2D(typ, txp2)] * (
	 (confy[coo2D(typ, txp2)]-confy[coo2D(typ, txp)])*(2*confz[coo2D(typ, txp2)]-confz[coo2D(ty, txp2)]-confz[coo2D(ty, txp)])
	-(confz[coo2D(typ, txp2)]-confz[coo2D(typ, txp)])*(2*confy[coo2D(typ, txp2)]-confy[coo2D(ty, txp2)]-confy[coo2D(ty, txp)])
	)+confy[coo2D(typ, txp2)]*(                                                                                            
	 (confz[coo2D(typ, txp2)]-confz[coo2D(typ, txp)])*(2*confx[coo2D(typ, txp2)]-confx[coo2D(ty, txp2)]-confx[coo2D(ty, txp)])
	-(confx[coo2D(typ, txp2)]-confx[coo2D(typ, txp)])*(2*confz[coo2D(typ, txp2)]-confz[coo2D(ty, txp2)]-confz[coo2D(ty, txp)])
	)+confz[coo2D(typ, txp2)] * (                                                                                          
	 (confx[coo2D(typ, txp2)]-confx[coo2D(typ, txp)])*(2*confy[coo2D(typ, txp2)]-confy[coo2D(ty, txp2)]-confy[coo2D(ty, txp)])
	-(confy[coo2D(typ, txp2)]-confy[coo2D(typ, txp)])*(2*confx[coo2D(typ, txp2)]-confx[coo2D(ty, txp2)]-confx[coo2D(ty, txp)])
	);
	//(2,2)
	sD[y][x] += confx[coo2D(typ2, txp2)] * (
	 (confy[coo2D(typ2, txp2)]-confy[coo2D(typ2, txp)])*(2*confz[coo2D(typ2, txp2)]-confz[coo2D(typ, txp2)]-confz[coo2D(typ, txp)])
	-(confz[coo2D(typ2, txp2)]-confz[coo2D(typ2, txp)])*(2*confy[coo2D(typ2, txp2)]-confy[coo2D(typ, txp2)]-confy[coo2D(typ, txp)])
	)+confy[coo2D(typ2, txp2)]*( 
	 (confz[coo2D(typ2, txp2)]-confz[coo2D(typ2, txp)])*(2*confx[coo2D(typ2, txp2)]-confx[coo2D(typ, txp2)]-confx[coo2D(typ, txp)])
	-(confx[coo2D(typ2, txp2)]-confx[coo2D(typ2, txp)])*(2*confz[coo2D(typ2, txp2)]-confz[coo2D(typ, txp2)]-confz[coo2D(typ, txp)])
	)+confz[coo2D(typ2, txp2)] * (
	 (confx[coo2D(typ2, txp2)]-confx[coo2D(typ2, txp)])*(2*confy[coo2D(typ2, txp2)]-confy[coo2D(typ, txp2)]-confy[coo2D(typ, txp)])
	-(confy[coo2D(typ2, txp2)]-confy[coo2D(typ2, txp)])*(2*confx[coo2D(typ2, txp2)]-confx[coo2D(typ, txp2)]-confx[coo2D(typ, txp)])
	);
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
