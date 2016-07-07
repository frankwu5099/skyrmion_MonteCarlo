#ifdef SQ

#include "measurements.cuh"
__global__ void cal2D(float *confx, float *confy, float *confz, double *out){
	//Energy variables
	extern __shared__ double sD[];
	const int x = threadIdx.x % (BlockSize_x);
	const int y = (threadIdx.x / BlockSize_x);
	const int tx = 2 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
	const int ty =(blockIdx.x / BN) * SpinSize +  2 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
	const int txp = tx +1 ;
	const int typ = ty +1 ;
	//const int ty = 2 * ((blockIdx.x / BN) * SpinSize + ((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
	const int dataoff = (blockIdx.x / BN) * MEASURE_NUM * BN;
	int bx, by;
	//-----Calculate the energy of each spin pairs in the system-----
	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.

	bx = (tx + SpinSize - 1) % SpinSize;
	if((ty % SpinSize) == 0)	by = ty + SpinSize - 1;
	else				by = ty - 1;
	//Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.

	//Top-left corner
	sD[y][x] = -confx[coo2D(ty, tx)] * ( BXMxx * confx[coo2D(ty, bx)] + BYMxx * confx[coo2D(by, tx)])\
	           -confx[coo2D(ty, tx)] * ( BXMxy * confy[coo2D(ty, bx)] + BYMxy * confy[coo2D(by, tx)])\
	           -confx[coo2D(ty, tx)] * ( BXMxz * confz[coo2D(ty, bx)] + BYMxz * confz[coo2D(by, tx)])\
		         -confy[coo2D(ty, tx)] * ( BXMyx * confx[coo2D(ty, bx)] + BYMyx * confx[coo2D(by, tx)])\
		         -confy[coo2D(ty, tx)] * ( BXMyy * confy[coo2D(ty, bx)] + BYMyy * confy[coo2D(by, tx)])\
		         -confy[coo2D(ty, tx)] * ( BXMyz * confz[coo2D(ty, bx)] + BYMyz * confz[coo2D(by, tx)])\
		         -confz[coo2D(ty, tx)] * ( BXMzx * confx[coo2D(ty, bx)] + BYMzx * confx[coo2D(by, tx)])\
		         -confz[coo2D(ty, tx)] * ( BXMzy * confy[coo2D(ty, bx)] + BYMzy * confy[coo2D(by, tx)])\
		         -confz[coo2D(ty, tx)] * ( BXMzz * confz[coo2D(ty, bx)] + BYMzz * confz[coo2D(by, tx)] - A * confz[coo2D(ty, tx)]);
	//Bottom-left corner
	sD[y][x] -= confx[coo2D((ty+1), tx)] * ( BXMxx * confx[coo2D((ty+1), bx)] + BYMxx * confx[coo2D(ty, tx)])\
		         +confx[coo2D((ty+1), tx)] * ( BXMxy * confy[coo2D((ty+1), bx)] + BYMxy * confy[coo2D(ty, tx)])\
		         +confx[coo2D((ty+1), tx)] * ( BXMxz * confz[coo2D((ty+1), bx)] + BYMxz * confz[coo2D(ty, tx)])\
		         +confy[coo2D((ty+1), tx)] * ( BXMyx * confx[coo2D((ty+1), bx)] + BYMyx * confx[coo2D(ty, tx)])\
		         +confy[coo2D((ty+1), tx)] * ( BXMyy * confy[coo2D((ty+1), bx)] + BYMyy * confy[coo2D(ty, tx)])\
		         +confy[coo2D((ty+1), tx)] * ( BXMyz * confz[coo2D((ty+1), bx)] + BYMyz * confz[coo2D(ty, tx)])\
		         +confz[coo2D((ty+1), tx)] * ( BXMzx * confx[coo2D((ty+1), bx)] + BYMzx * confx[coo2D(ty, tx)])\
		         +confz[coo2D((ty+1), tx)] * ( BXMzy * confy[coo2D((ty+1), bx)] + BYMzy * confy[coo2D(ty, tx)])\
		         +confz[coo2D((ty+1), tx)] * ( BXMzz * confz[coo2D((ty+1), bx)] + BYMzz * confz[coo2D(ty, tx)] - A * confz[coo2D((ty+1), tx)]);
	//Top-right corner
	sD[y][x] -= confx[coo2D(ty, tx+1)] * ( BXMxx * confx[coo2D(ty, tx)] + BYMxx * confx[coo2D(by, tx+1)])\
		         +confx[coo2D(ty, tx+1)] * ( BXMxy * confy[coo2D(ty, tx)] + BYMxy * confy[coo2D(by, tx+1)])\
		         +confx[coo2D(ty, tx+1)] * ( BXMxz * confz[coo2D(ty, tx)] + BYMxz * confz[coo2D(by, tx+1)])\
		         +confy[coo2D(ty, tx+1)] * ( BXMyx * confx[coo2D(ty, tx)] + BYMyx * confx[coo2D(by, tx+1)])\
		         +confy[coo2D(ty, tx+1)] * ( BXMyy * confy[coo2D(ty, tx)] + BYMyy * confy[coo2D(by, tx+1)])\
		         +confy[coo2D(ty, tx+1)] * ( BXMyz * confz[coo2D(ty, tx)] + BYMyz * confz[coo2D(by, tx+1)])\
		         +confz[coo2D(ty, tx+1)] * ( BXMzx * confx[coo2D(ty, tx)] + BYMzx * confx[coo2D(by, tx+1)])\
		         +confz[coo2D(ty, tx+1)] * ( BXMzy * confy[coo2D(ty, tx)] + BYMzy * confy[coo2D(by, tx+1)])\
		         +confz[coo2D(ty, tx+1)] * ( BXMzz * confz[coo2D(ty, tx)] + BYMzz * confz[coo2D(by, tx+1)] - A * confz[coo2D(ty, tx+1)]);
	//Bottom-right corner
	sD[y][x] -= confx[coo2D((ty+1), tx+1)] * ( BXMxx * confx[coo2D((ty+1), tx)] + BYMxx * confx[coo2D(ty, tx+1)])\
		         +confx[coo2D((ty+1), tx+1)] * ( BXMxy * confy[coo2D((ty+1), tx)] + BYMxy * confy[coo2D(ty, tx+1)])\
		         +confx[coo2D((ty+1), tx+1)] * ( BXMxz * confz[coo2D((ty+1), tx)] + BYMxz * confz[coo2D(ty, tx+1)])\
		         +confy[coo2D((ty+1), tx+1)] * ( BXMyx * confx[coo2D((ty+1), tx)] + BYMyx * confx[coo2D(ty, tx+1)])\
		         +confy[coo2D((ty+1), tx+1)] * ( BXMyy * confy[coo2D((ty+1), tx)] + BYMyy * confy[coo2D(ty, tx+1)])\
		         +confy[coo2D((ty+1), tx+1)] * ( BXMyz * confz[coo2D((ty+1), tx)] + BYMyz * confz[coo2D(ty, tx+1)])\
		         +confz[coo2D((ty+1), tx+1)] * ( BXMzx * confx[coo2D((ty+1), tx)] + BYMzx * confx[coo2D(ty, tx+1)])\
		         +confz[coo2D((ty+1), tx+1)] * ( BXMzy * confy[coo2D((ty+1), tx)] + BYMzy * confy[coo2D(ty, tx+1)])\
		         +confz[coo2D((ty+1), tx+1)] * ( BXMzz * confz[coo2D((ty+1), tx)] + BYMzz * confz[coo2D(ty, tx+1)] - A * confz[coo2D((ty+1), tx+1)]);
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
	sD[y][x] += confx[coo2D((ty+1), tx)];
	sD[y][x] += confx[coo2D(ty, tx+1)];
	sD[y][x] += confx[coo2D((ty+1), (tx+1))];
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
	sD[y][x]  = confy[coo2D(ty, tx)];
	//Bottom-left corner
	sD[y][x] += confy[coo2D((ty+1), tx)];
	//Top-right corner
	sD[y][x] += confy[coo2D(ty, tx+1)];
	//Bottom-right corner
	sD[y][x] += confy[coo2D((ty+1), tx+1)];
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
	//Top-left corner
	sD[y][x]  = confz[coo2D(ty, tx)];
	//Bottom-left corner
	sD[y][x] += confz[coo2D((ty+1), tx)];
	//Top-right corner
	sD[y][x] += confz[coo2D(ty, tx+1)];
	//Bottom-right corner
	sD[y][x] += confz[coo2D((ty+1), tx+1)];
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
	sD[y][x]  = confx[coo2D(ty, tx)] * (
	(confy[coo2D(ty, tx)]-confy[coo2D(ty, bx)])*(confz[coo2D(ty, tx)]-confz[coo2D(by, tx)])
	-(confz[coo2D(ty, tx)]-confz[coo2D(ty, bx)])*(confy[coo2D(ty, tx)]-confy[coo2D(by, tx)])
	)+confy[coo2D(ty, tx)] * (
	(confz[coo2D(ty, tx)]-confz[coo2D(ty, bx)])*(confx[coo2D(ty, tx)]-confx[coo2D(by, tx)])
	-(confx[coo2D(ty, tx)]-confx[coo2D(ty, bx)])*(confz[coo2D(ty, tx)]-confz[coo2D(by, tx)])
	)+confz[coo2D(ty, tx)] * (
	(confx[coo2D(ty, tx)]-confx[coo2D(ty, bx)])*(confy[coo2D(ty, tx)]-confy[coo2D(by, tx)])
	-(confy[coo2D(ty, tx)]-confy[coo2D(ty, bx)])*(confx[coo2D(ty, tx)]-confx[coo2D(by, tx)])
	);
	//Bottom-left corner
	sD[y][x] += confx[coo2D(typ, tx)] * (
	(confy[coo2D(typ, tx)]-confy[coo2D(typ, bx)])*(confz[coo2D(typ, tx)]-confz[coo2D(ty, tx)])
	-(confz[coo2D(typ, tx)]-confz[coo2D(typ, bx)])*(confy[coo2D(typ, tx)]-confy[coo2D(ty, tx)])
	)+confy[coo2D(typ, tx)]*(
	(confz[coo2D(typ, tx)]-confz[coo2D(typ, bx)])*(confx[coo2D(typ, tx)]-confx[coo2D(ty, tx)])
	-(confx[coo2D(typ, tx)]-confx[coo2D(typ, bx)])*(confz[coo2D(typ, tx)]-confz[coo2D(ty, tx)])
	)+confz[coo2D(typ, tx)] * (
	(confx[coo2D(typ, tx)]-confx[coo2D(typ, bx)])*(confy[coo2D(typ, tx)]-confy[coo2D(ty, tx)])
	-(confy[coo2D(typ, tx)]-confy[coo2D(typ, bx)])*(confx[coo2D(typ, tx)]-confx[coo2D(ty, tx)])
	);
	//Top-right corner
	sD[y][x] += confx[coo2D(ty, txp)] * (
	(confy[coo2D(ty, txp)]-confy[coo2D(ty, tx)])*(confz[coo2D(ty, txp)]-confz[coo2D(by, txp)])
	-(confz[coo2D(ty, txp)]-confz[coo2D(ty, tx)])*(confy[coo2D(ty, txp)]-confy[coo2D(by, txp)])
	)+confy[coo2D(ty, txp)]*(
	(confz[coo2D(ty, txp)]-confz[coo2D(ty, tx)])*(confx[coo2D(ty, txp)]-confx[coo2D(by, txp)])
	-(confx[coo2D(ty, txp)]-confx[coo2D(ty, tx)])*(confz[coo2D(ty, txp)]-confz[coo2D(by, txp)])
	)+confz[coo2D(ty, txp)] * (
	(confx[coo2D(ty, txp)]-confx[coo2D(ty, tx)])*(confy[coo2D(ty, txp)]-confy[coo2D(by, txp)])
	-(confy[coo2D(ty, txp)]-confy[coo2D(ty, tx)])*(confx[coo2D(ty, txp)]-confx[coo2D(by, txp)])
	);
	//Bottom-right corner
	sD[y][x] += confx[coo2D(typ, txp)] * (
	(confy[coo2D(typ, txp)]-confy[coo2D(typ, tx)])*(confz[coo2D(typ, txp)]-confz[coo2D(ty, txp)])
	-(confz[coo2D(typ, txp)]-confz[coo2D(typ, tx)])*(confy[coo2D(typ, txp)]-confy[coo2D(ty, txp)])
	)+confy[coo2D(typ, txp)]*(
	(confz[coo2D(typ, txp)]-confz[coo2D(typ, tx)])*(confx[coo2D(typ, txp)]-confx[coo2D(ty, txp)])
	-(confx[coo2D(typ, txp)]-confx[coo2D(typ, tx)])*(confz[coo2D(typ, txp)]-confz[coo2D(ty, txp)])
	)+confz[coo2D(typ, txp)] * (
	(confx[coo2D(typ, txp)]-confx[coo2D(typ, tx)])*(confy[coo2D(typ, txp)]-confy[coo2D(ty, txp)])
	-(confy[coo2D(typ, txp)]-confy[coo2D(typ, tx)])*(confx[coo2D(typ, txp)]-confx[coo2D(ty, txp)])
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
