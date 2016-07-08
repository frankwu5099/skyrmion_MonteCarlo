#include "measurements.cuh"
#ifdef TRI
__constant__ unsigned int cals_SpinSize;
__constant__ unsigned int cals_SpinSize_z;
__constant__ unsigned int cals_BlockSize_x;
__constant__ unsigned int cals_BlockSize_y;
__constant__ unsigned int cals_GridSize_x;
__constant__ unsigned int cals_GridSize_y;
__constant__ unsigned int cals_TN;
__constant__ unsigned int cals_BN;
__constant__ float cals_A; //(0.0)
void move_params_device_cals(){
  cudaMemcpyToSymbol( cals_SpinSize, &H_SpinSize, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_SpinSize_z, &H_SpinSize_z, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BlockSize_x, &H_BlockSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BlockSize_y, &H_BlockSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_GridSize_x, &H_GridSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_GridSize_y, &H_GridSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_TN, &H_TN, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BN, &H_BN, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_A , &H_A , sizeof(float));
}
__global__ void calTRI(float *confx, float *confy, float *confz, double *out){
	//Energy variables
	extern __shared__ double sD[];
	const int x = threadIdx.x % (cals_BlockSize_x);
	const int y = (threadIdx.x / cals_BlockSize_x);
	const int tx = 3 * (((blockIdx.x % cals_BN) % cals_GridSize_x) * cals_BlockSize_x + x);
	const int ty =(blockIdx.x / cals_BN) * cals_SpinSize +  3 * ((((blockIdx.x % cals_BN) / cals_GridSize_x) % cals_GridSize_y) * cals_BlockSize_y + y);
	const int txp = tx +1 ;
	const int typ = ty +1 ;
	const int txp2 = tx +2 ;
	const int typ2 = ty +2 ;
	//const int ty = 2 * ((blockIdx.x / cals_BN) * cals_SpinSize + ((blockIdx.x % cals_BN) / cals_GridSize_x) * cals_BlockSize_y + y);
	const int dataoff = (blockIdx.x / cals_BN) * MEASURE_NUM * cals_BN;
	int bx, by;
	//-----Calculate the energy of each spin pairs in the system-----
	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.

	bx = (tx + cals_SpinSize - 1) % cals_SpinSize;
	if((ty % cals_SpinSize) == 0)	by = ty + cals_SpinSize - 1;
	else				by = ty - 1;
	//Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.

	//0,0
	sD[threadIdx.x] = -confx[cals_coo2D(ty, tx)] * ( BXMxx * confx[cals_coo2D(ty, bx)] + BYMxx * confx[cals_coo2D(by, tx)] + BWMxx * confx[cals_coo2D(by, bx)])\
	           -confx[cals_coo2D(ty, tx)] * ( BXMxy * confy[cals_coo2D(ty, bx)] + BYMxy * confy[cals_coo2D(by, tx)] + BWMxy * confy[cals_coo2D(by, bx)])\
	           -confx[cals_coo2D(ty, tx)] * ( BXMxz * confz[cals_coo2D(ty, bx)] + BYMxz * confz[cals_coo2D(by, tx)] + BWMxz * confz[cals_coo2D(by, bx)])\
		         -confy[cals_coo2D(ty, tx)] * ( BXMyx * confx[cals_coo2D(ty, bx)] + BYMyx * confx[cals_coo2D(by, tx)] + BWMyx * confx[cals_coo2D(by, bx)])\
		         -confy[cals_coo2D(ty, tx)] * ( BXMyy * confy[cals_coo2D(ty, bx)] + BYMyy * confy[cals_coo2D(by, tx)] + BWMyy * confy[cals_coo2D(by, bx)])\
		         -confy[cals_coo2D(ty, tx)] * ( BXMyz * confz[cals_coo2D(ty, bx)] + BYMyz * confz[cals_coo2D(by, tx)] + BWMyz * confz[cals_coo2D(by, bx)])\
		         -confz[cals_coo2D(ty, tx)] * ( BXMzx * confx[cals_coo2D(ty, bx)] + BYMzx * confx[cals_coo2D(by, tx)] + BWMzx * confx[cals_coo2D(by, bx)])\
		         -confz[cals_coo2D(ty, tx)] * ( BXMzy * confy[cals_coo2D(ty, bx)] + BYMzy * confy[cals_coo2D(by, tx)] + BWMzy * confy[cals_coo2D(by, bx)])\
		         -confz[cals_coo2D(ty, tx)] * ( BXMzz * confz[cals_coo2D(ty, bx)] + BYMzz * confz[cals_coo2D(by, tx)] + BWMzz * confz[cals_coo2D(by, bx)] - cals_A * confz[cals_coo2D(ty, tx)]);
	//1,0
	sD[threadIdx.x] -= confx[cals_coo2D(typ, tx)] * ( BXMxx * confx[cals_coo2D(typ, bx)] + BYMxx * confx[cals_coo2D(ty, tx)] + BWMxx * confx[cals_coo2D(ty, bx)])\
		         +confx[cals_coo2D(typ, tx)] * ( BXMxy * confy[cals_coo2D(typ, bx)] + BYMxy * confy[cals_coo2D(ty, tx)] + BWMxy * confy[cals_coo2D(ty, bx)])\
		         +confx[cals_coo2D(typ, tx)] * ( BXMxz * confz[cals_coo2D(typ, bx)] + BYMxz * confz[cals_coo2D(ty, tx)] + BWMxz * confz[cals_coo2D(ty, bx)])\
		         +confy[cals_coo2D(typ, tx)] * ( BXMyx * confx[cals_coo2D(typ, bx)] + BYMyx * confx[cals_coo2D(ty, tx)] + BWMyx * confx[cals_coo2D(ty, bx)])\
		         +confy[cals_coo2D(typ, tx)] * ( BXMyy * confy[cals_coo2D(typ, bx)] + BYMyy * confy[cals_coo2D(ty, tx)] + BWMyy * confy[cals_coo2D(ty, bx)])\
		         +confy[cals_coo2D(typ, tx)] * ( BXMyz * confz[cals_coo2D(typ, bx)] + BYMyz * confz[cals_coo2D(ty, tx)] + BWMyz * confz[cals_coo2D(ty, bx)])\
		         +confz[cals_coo2D(typ, tx)] * ( BXMzx * confx[cals_coo2D(typ, bx)] + BYMzx * confx[cals_coo2D(ty, tx)] + BWMzx * confx[cals_coo2D(ty, bx)])\
		         +confz[cals_coo2D(typ, tx)] * ( BXMzy * confy[cals_coo2D(typ, bx)] + BYMzy * confy[cals_coo2D(ty, tx)] + BWMzy * confy[cals_coo2D(ty, bx)])\
		         +confz[cals_coo2D(typ, tx)] * ( BXMzz * confz[cals_coo2D(typ, bx)] + BYMzz * confz[cals_coo2D(ty, tx)] + BWMzz * confz[cals_coo2D(ty, bx)] - cals_A * confz[cals_coo2D((ty+1), tx)]);
	//2,0
	sD[threadIdx.x] -= confx[cals_coo2D(typ2, tx)] * ( BXMxx * confx[cals_coo2D(typ2, bx)] + BYMxx * confx[cals_coo2D(typ, tx)] + BWMxx * confx[cals_coo2D(typ, bx)])\
		         +confx[cals_coo2D(typ2, tx)] * ( BXMxy * confy[cals_coo2D(typ2, bx)] + BYMxy * confy[cals_coo2D(typ, tx)] + BWMxy * confy[cals_coo2D(typ, bx)])\
		         +confx[cals_coo2D(typ2, tx)] * ( BXMxz * confz[cals_coo2D(typ2, bx)] + BYMxz * confz[cals_coo2D(typ, tx)] + BWMxz * confz[cals_coo2D(typ, bx)])\
		         +confy[cals_coo2D(typ2, tx)] * ( BXMyx * confx[cals_coo2D(typ2, bx)] + BYMyx * confx[cals_coo2D(typ, tx)] + BWMyx * confx[cals_coo2D(typ, bx)])\
		         +confy[cals_coo2D(typ2, tx)] * ( BXMyy * confy[cals_coo2D(typ2, bx)] + BYMyy * confy[cals_coo2D(typ, tx)] + BWMyy * confy[cals_coo2D(typ, bx)])\
		         +confy[cals_coo2D(typ2, tx)] * ( BXMyz * confz[cals_coo2D(typ2, bx)] + BYMyz * confz[cals_coo2D(typ, tx)] + BWMyz * confz[cals_coo2D(typ, bx)])\
		         +confz[cals_coo2D(typ2, tx)] * ( BXMzx * confx[cals_coo2D(typ2, bx)] + BYMzx * confx[cals_coo2D(typ, tx)] + BWMzx * confx[cals_coo2D(typ, bx)])\
		         +confz[cals_coo2D(typ2, tx)] * ( BXMzy * confy[cals_coo2D(typ2, bx)] + BYMzy * confy[cals_coo2D(typ, tx)] + BWMzy * confy[cals_coo2D(typ, bx)])\
		         +confz[cals_coo2D(typ2, tx)] * ( BXMzz * confz[cals_coo2D(typ2, bx)] + BYMzz * confz[cals_coo2D(typ, tx)] + BWMzz * confz[cals_coo2D(typ, bx)] - cals_A * confz[cals_coo2D((ty+1), tx)]);
	//0,1
	sD[threadIdx.x] -= confx[cals_coo2D(ty, txp)] * ( BXMxx * confx[cals_coo2D(ty, tx)] + BYMxx * confx[cals_coo2D(by, txp)] + BWMxx * confx[cals_coo2D(by, tx)])\
		         +confx[cals_coo2D(ty, txp)] * ( BXMxy * confy[cals_coo2D(ty, tx)] + BYMxy * confy[cals_coo2D(by, txp)] + BWMxy * confy[cals_coo2D(by, tx)])\
		         +confx[cals_coo2D(ty, txp)] * ( BXMxz * confz[cals_coo2D(ty, tx)] + BYMxz * confz[cals_coo2D(by, txp)] + BWMxz * confz[cals_coo2D(by, tx)])\
		         +confy[cals_coo2D(ty, txp)] * ( BXMyx * confx[cals_coo2D(ty, tx)] + BYMyx * confx[cals_coo2D(by, txp)] + BWMyx * confx[cals_coo2D(by, tx)])\
		         +confy[cals_coo2D(ty, txp)] * ( BXMyy * confy[cals_coo2D(ty, tx)] + BYMyy * confy[cals_coo2D(by, txp)] + BWMyy * confy[cals_coo2D(by, tx)])\
		         +confy[cals_coo2D(ty, txp)] * ( BXMyz * confz[cals_coo2D(ty, tx)] + BYMyz * confz[cals_coo2D(by, txp)] + BWMyz * confz[cals_coo2D(by, tx)])\
		         +confz[cals_coo2D(ty, txp)] * ( BXMzx * confx[cals_coo2D(ty, tx)] + BYMzx * confx[cals_coo2D(by, txp)] + BWMzx * confx[cals_coo2D(by, tx)])\
		         +confz[cals_coo2D(ty, txp)] * ( BXMzy * confy[cals_coo2D(ty, tx)] + BYMzy * confy[cals_coo2D(by, txp)] + BWMzy * confy[cals_coo2D(by, tx)])\
		         +confz[cals_coo2D(ty, txp)] * ( BXMzz * confz[cals_coo2D(ty, tx)] + BYMzz * confz[cals_coo2D(by, txp)] + BWMzz * confz[cals_coo2D(by, tx)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//1,1
	sD[threadIdx.x] -= confx[cals_coo2D(typ, txp)] * ( BXMxx * confx[cals_coo2D(typ, tx)] + BYMxx * confx[cals_coo2D(ty, txp)] + BWMxx * confx[cals_coo2D(ty, tx)])\
		         +confx[cals_coo2D(typ, txp)] * ( BXMxy * confy[cals_coo2D(typ, tx)] + BYMxy * confy[cals_coo2D(ty, txp)] + BWMxy * confy[cals_coo2D(ty, tx)])\
		         +confx[cals_coo2D(typ, txp)] * ( BXMxz * confz[cals_coo2D(typ, tx)] + BYMxz * confz[cals_coo2D(ty, txp)] + BWMxz * confz[cals_coo2D(ty, tx)])\
		         +confy[cals_coo2D(typ, txp)] * ( BXMyx * confx[cals_coo2D(typ, tx)] + BYMyx * confx[cals_coo2D(ty, txp)] + BWMyx * confx[cals_coo2D(ty, tx)])\
		         +confy[cals_coo2D(typ, txp)] * ( BXMyy * confy[cals_coo2D(typ, tx)] + BYMyy * confy[cals_coo2D(ty, txp)] + BWMyy * confy[cals_coo2D(ty, tx)])\
		         +confy[cals_coo2D(typ, txp)] * ( BXMyz * confz[cals_coo2D(typ, tx)] + BYMyz * confz[cals_coo2D(ty, txp)] + BWMyz * confz[cals_coo2D(ty, tx)])\
		         +confz[cals_coo2D(typ, txp)] * ( BXMzx * confx[cals_coo2D(typ, tx)] + BYMzx * confx[cals_coo2D(ty, txp)] + BWMzx * confx[cals_coo2D(ty, tx)])\
		         +confz[cals_coo2D(typ, txp)] * ( BXMzy * confy[cals_coo2D(typ, tx)] + BYMzy * confy[cals_coo2D(ty, txp)] + BWMzy * confy[cals_coo2D(ty, tx)])\
		         +confz[cals_coo2D(typ, txp)] * ( BXMzz * confz[cals_coo2D(typ, tx)] + BYMzz * confz[cals_coo2D(ty, txp)] + BWMzz * confz[cals_coo2D(ty, tx)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//2,1
	sD[threadIdx.x] -= confx[cals_coo2D(typ2, txp)] * ( BXMxx * confx[cals_coo2D(typ2, tx)] + BYMxx * confx[cals_coo2D(typ, txp)] + BWMxx * confx[cals_coo2D(typ, tx)])\
		         +confx[cals_coo2D(typ2, txp)] * ( BXMxy * confy[cals_coo2D(typ2, tx)] + BYMxy * confy[cals_coo2D(typ, txp)] + BWMxy * confy[cals_coo2D(typ, tx)])\
		         +confx[cals_coo2D(typ2, txp)] * ( BXMxz * confz[cals_coo2D(typ2, tx)] + BYMxz * confz[cals_coo2D(typ, txp)] + BWMxz * confz[cals_coo2D(typ, tx)])\
		         +confy[cals_coo2D(typ2, txp)] * ( BXMyx * confx[cals_coo2D(typ2, tx)] + BYMyx * confx[cals_coo2D(typ, txp)] + BWMyx * confx[cals_coo2D(typ, tx)])\
		         +confy[cals_coo2D(typ2, txp)] * ( BXMyy * confy[cals_coo2D(typ2, tx)] + BYMyy * confy[cals_coo2D(typ, txp)] + BWMyy * confy[cals_coo2D(typ, tx)])\
		         +confy[cals_coo2D(typ2, txp)] * ( BXMyz * confz[cals_coo2D(typ2, tx)] + BYMyz * confz[cals_coo2D(typ, txp)] + BWMyz * confz[cals_coo2D(typ, tx)])\
		         +confz[cals_coo2D(typ2, txp)] * ( BXMzx * confx[cals_coo2D(typ2, tx)] + BYMzx * confx[cals_coo2D(typ, txp)] + BWMzx * confx[cals_coo2D(typ, tx)])\
		         +confz[cals_coo2D(typ2, txp)] * ( BXMzy * confy[cals_coo2D(typ2, tx)] + BYMzy * confy[cals_coo2D(typ, txp)] + BWMzy * confy[cals_coo2D(typ, tx)])\
		         +confz[cals_coo2D(typ2, txp)] * ( BXMzz * confz[cals_coo2D(typ2, tx)] + BYMzz * confz[cals_coo2D(typ, txp)] + BWMzz * confz[cals_coo2D(typ, tx)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//0,2
	sD[threadIdx.x] -= confx[cals_coo2D(ty, txp2)] * ( BXMxx * confx[cals_coo2D(ty, txp)] + BYMxx * confx[cals_coo2D(by, txp2)] + BWMxx * confx[cals_coo2D(by, txp)])\
		         +confx[cals_coo2D(ty, txp2)] * ( BXMxy * confy[cals_coo2D(ty, txp)] + BYMxy * confy[cals_coo2D(by, txp2)] + BWMxy * confy[cals_coo2D(by, txp)])\
		         +confx[cals_coo2D(ty, txp2)] * ( BXMxz * confz[cals_coo2D(ty, txp)] + BYMxz * confz[cals_coo2D(by, txp2)] + BWMxz * confz[cals_coo2D(by, txp)])\
		         +confy[cals_coo2D(ty, txp2)] * ( BXMyx * confx[cals_coo2D(ty, txp)] + BYMyx * confx[cals_coo2D(by, txp2)] + BWMyx * confx[cals_coo2D(by, txp)])\
		         +confy[cals_coo2D(ty, txp2)] * ( BXMyy * confy[cals_coo2D(ty, txp)] + BYMyy * confy[cals_coo2D(by, txp2)] + BWMyy * confy[cals_coo2D(by, txp)])\
		         +confy[cals_coo2D(ty, txp2)] * ( BXMyz * confz[cals_coo2D(ty, txp)] + BYMyz * confz[cals_coo2D(by, txp2)] + BWMyz * confz[cals_coo2D(by, txp)])\
		         +confz[cals_coo2D(ty, txp2)] * ( BXMzx * confx[cals_coo2D(ty, txp)] + BYMzx * confx[cals_coo2D(by, txp2)] + BWMzx * confx[cals_coo2D(by, txp)])\
		         +confz[cals_coo2D(ty, txp2)] * ( BXMzy * confy[cals_coo2D(ty, txp)] + BYMzy * confy[cals_coo2D(by, txp2)] + BWMzy * confy[cals_coo2D(by, txp)])\
		         +confz[cals_coo2D(ty, txp2)] * ( BXMzz * confz[cals_coo2D(ty, txp)] + BYMzz * confz[cals_coo2D(by, txp2)] + BWMzz * confz[cals_coo2D(by, txp)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//1,2
	sD[threadIdx.x] -= confx[cals_coo2D(typ, txp2)] * ( BXMxx * confx[cals_coo2D(typ, txp)] + BYMxx * confx[cals_coo2D(ty, txp2)] + BWMxx * confx[cals_coo2D(ty, txp)])\
		         +confx[cals_coo2D(typ, txp2)] * ( BXMxy * confy[cals_coo2D(typ, txp)] + BYMxy * confy[cals_coo2D(ty, txp2)] + BWMxy * confy[cals_coo2D(ty, txp)])\
		         +confx[cals_coo2D(typ, txp2)] * ( BXMxz * confz[cals_coo2D(typ, txp)] + BYMxz * confz[cals_coo2D(ty, txp2)] + BWMxz * confz[cals_coo2D(ty, txp)])\
		         +confy[cals_coo2D(typ, txp2)] * ( BXMyx * confx[cals_coo2D(typ, txp)] + BYMyx * confx[cals_coo2D(ty, txp2)] + BWMyx * confx[cals_coo2D(ty, txp)])\
		         +confy[cals_coo2D(typ, txp2)] * ( BXMyy * confy[cals_coo2D(typ, txp)] + BYMyy * confy[cals_coo2D(ty, txp2)] + BWMyy * confy[cals_coo2D(ty, txp)])\
		         +confy[cals_coo2D(typ, txp2)] * ( BXMyz * confz[cals_coo2D(typ, txp)] + BYMyz * confz[cals_coo2D(ty, txp2)] + BWMyz * confz[cals_coo2D(ty, txp)])\
		         +confz[cals_coo2D(typ, txp2)] * ( BXMzx * confx[cals_coo2D(typ, txp)] + BYMzx * confx[cals_coo2D(ty, txp2)] + BWMzx * confx[cals_coo2D(ty, txp)])\
		         +confz[cals_coo2D(typ, txp2)] * ( BXMzy * confy[cals_coo2D(typ, txp)] + BYMzy * confy[cals_coo2D(ty, txp2)] + BWMzy * confy[cals_coo2D(ty, txp)])\
		         +confz[cals_coo2D(typ, txp2)] * ( BXMzz * confz[cals_coo2D(typ, txp)] + BYMzz * confz[cals_coo2D(ty, txp2)] + BWMzz * confz[cals_coo2D(ty, txp)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//2,2
	sD[threadIdx.x] -= confx[cals_coo2D(typ2, txp2)] * ( BXMxx * confx[cals_coo2D(typ2, txp)] + BYMxx * confx[cals_coo2D(typ, txp2)] + BWMxx * confx[cals_coo2D(typ, txp)])\
		         +confx[cals_coo2D(typ2, txp2)] * ( BXMxy * confy[cals_coo2D(typ2, txp)] + BYMxy * confy[cals_coo2D(typ, txp2)] + BWMxy * confy[cals_coo2D(typ, txp)])\
		         +confx[cals_coo2D(typ2, txp2)] * ( BXMxz * confz[cals_coo2D(typ2, txp)] + BYMxz * confz[cals_coo2D(typ, txp2)] + BWMxz * confz[cals_coo2D(typ, txp)])\
		         +confy[cals_coo2D(typ2, txp2)] * ( BXMyx * confx[cals_coo2D(typ2, txp)] + BYMyx * confx[cals_coo2D(typ, txp2)] + BWMyx * confx[cals_coo2D(typ, txp)])\
		         +confy[cals_coo2D(typ2, txp2)] * ( BXMyy * confy[cals_coo2D(typ2, txp)] + BYMyy * confy[cals_coo2D(typ, txp2)] + BWMyy * confy[cals_coo2D(typ, txp)])\
		         +confy[cals_coo2D(typ2, txp2)] * ( BXMyz * confz[cals_coo2D(typ2, txp)] + BYMyz * confz[cals_coo2D(typ, txp2)] + BWMyz * confz[cals_coo2D(typ, txp)])\
		         +confz[cals_coo2D(typ2, txp2)] * ( BXMzx * confx[cals_coo2D(typ2, txp)] + BYMzx * confx[cals_coo2D(typ, txp2)] + BWMzx * confx[cals_coo2D(typ, txp)])\
		         +confz[cals_coo2D(typ2, txp2)] * ( BXMzy * confy[cals_coo2D(typ2, txp)] + BYMzy * confy[cals_coo2D(typ, txp2)] + BWMzy * confy[cals_coo2D(typ, txp)])\
		         +confz[cals_coo2D(typ2, txp2)] * ( BXMzz * confz[cals_coo2D(typ2, txp)] + BYMzz * confz[cals_coo2D(typ, txp2)] + BWMzz * confz[cals_coo2D(typ, txp)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	__syncthreads();


	//Sum over all elements in each sD
	if(cals_TN>=512){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>=256){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>=128){
		if((threadIdx.x < 64) && (threadIdx.x+64 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+64];
		}
		__syncthreads();
	}
	if(threadIdx.x < 32){
		sD[threadIdx.x] += sD[threadIdx.x+32];
	}
	__syncthreads();
	if(threadIdx.x < 16){
		sD[threadIdx.x] += sD[threadIdx.x+16];
	}
	__syncthreads();
	if(threadIdx.x < 8){
		sD[threadIdx.x] += sD[threadIdx.x+8];
	}
	__syncthreads();
	if(threadIdx.x < 4){
		sD[threadIdx.x] += sD[threadIdx.x+4];
	}
	__syncthreads();
	if(threadIdx.x < 2){
		sD[threadIdx.x] += sD[threadIdx.x+2];
	}
	__syncthreads();
	if(threadIdx.x < 1){
		sD[threadIdx.x] += sD[threadIdx.x+1];
	}
	__syncthreads();
	if(threadIdx.x == 0)
		out[dataoff + (blockIdx.x % cals_BN)] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confx[cals_coo2D(ty, tx)];
	sD[threadIdx.x] += confx[cals_coo2D(typ, tx)];
	sD[threadIdx.x] += confx[cals_coo2D(typ2, tx)];
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp)];
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp)];
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp)];
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp2)];
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp2)];
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp2)];
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>=512){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>=256){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>=128){
		if((threadIdx.x < 64) && (threadIdx.x+64 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+64];
		}
		__syncthreads();
	}
	if(threadIdx.x < 32){
		sD[threadIdx.x] += sD[threadIdx.x+32];
	}
	__syncthreads();
	if(threadIdx.x < 16){
		sD[threadIdx.x] += sD[threadIdx.x+16];
	}
	__syncthreads();
	if(threadIdx.x < 8){
		sD[threadIdx.x] += sD[threadIdx.x+8];
	}
	__syncthreads();
	if(threadIdx.x < 4){
		sD[threadIdx.x] += sD[threadIdx.x+4];
	}
	__syncthreads();
	if(threadIdx.x < 2){
		sD[threadIdx.x] += sD[threadIdx.x+2];
	}
	__syncthreads();
	if(threadIdx.x < 1){
		sD[threadIdx.x] += sD[threadIdx.x+1];
	}
	__syncthreads();
	if(threadIdx.x == 0)
		out[dataoff + (blockIdx.x % cals_BN) + cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in y direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confy[cals_coo2D(ty, tx)];
	sD[threadIdx.x] += confy[cals_coo2D(typ, tx)];
	sD[threadIdx.x] += confy[cals_coo2D(typ2, tx)];
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp)];
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp)];
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp)];
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp2)];
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp2)];
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp2)];
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>=512){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>=256){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>=128){
		if((threadIdx.x < 64) && (threadIdx.x+64 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+64];
		}
		__syncthreads();
	}
	if(threadIdx.x < 32){
		sD[threadIdx.x] += sD[threadIdx.x+32];
	}
	__syncthreads();
	if(threadIdx.x < 16){
		sD[threadIdx.x] += sD[threadIdx.x+16];
	}
	__syncthreads();
	if(threadIdx.x < 8){
		sD[threadIdx.x] += sD[threadIdx.x+8];
	}
	__syncthreads();
	if(threadIdx.x < 4){
		sD[threadIdx.x] += sD[threadIdx.x+4];
	}
	__syncthreads();
	if(threadIdx.x < 2){
		sD[threadIdx.x] += sD[threadIdx.x+2];
	}
	__syncthreads();
	if(threadIdx.x < 1){
		sD[threadIdx.x] += sD[threadIdx.x+1];
	}
	__syncthreads();
	if(threadIdx.x == 0)
		out[dataoff + (blockIdx.x % cals_BN) + 2*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in z direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confy[cals_coo2D(ty, tx)];
	sD[threadIdx.x] += confy[cals_coo2D(typ, tx)];
	sD[threadIdx.x] += confy[cals_coo2D(typ2, tx)];
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp)];
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp)];
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp)];
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp2)];
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp2)];
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp2)];
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>=512){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>=256){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>=128){
		if((threadIdx.x < 64) && (threadIdx.x+64 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+64];
		}
		__syncthreads();
	}
	if(threadIdx.x < 32){
		sD[threadIdx.x] += sD[threadIdx.x+32];
	}
	__syncthreads();
	if(threadIdx.x < 16){
		sD[threadIdx.x] += sD[threadIdx.x+16];
	}
	__syncthreads();
	if(threadIdx.x < 8){
		sD[threadIdx.x] += sD[threadIdx.x+8];
	}
	__syncthreads();
	if(threadIdx.x < 4){
		sD[threadIdx.x] += sD[threadIdx.x+4];
	}
	__syncthreads();
	if(threadIdx.x < 2){
		sD[threadIdx.x] += sD[threadIdx.x+2];
	}
	__syncthreads();
	if(threadIdx.x < 1){
		sD[threadIdx.x] += sD[threadIdx.x+1];
	}
	__syncthreads();
	if(threadIdx.x == 0)
		out[dataoff + (blockIdx.x % cals_BN) + 3*cals_BN] = sD[0];
	__syncthreads();

	//try to measure Chern number
	//(0,0)
	sD[threadIdx.x]  = confx[cals_coo2D(ty, tx)] * (
	 (confy[cals_coo2D(ty, tx)]-confy[cals_coo2D(ty, bx)])*(2*confz[cals_coo2D(ty, tx)]-confz[cals_coo2D(by, tx)]-confz[cals_coo2D(by, bx)])
	-(confz[cals_coo2D(ty, tx)]-confz[cals_coo2D(ty, bx)])*(2*confy[cals_coo2D(ty, tx)]-confy[cals_coo2D(by, tx)]-confy[cals_coo2D(by, bx)])
	)+confy[cals_coo2D(ty, tx)] * (
	 (confz[cals_coo2D(ty, tx)]-confz[cals_coo2D(ty, bx)])*(2*confx[cals_coo2D(ty, tx)]-confx[cals_coo2D(by, tx)]-confx[cals_coo2D(by, bx)])
	-(confx[cals_coo2D(ty, tx)]-confx[cals_coo2D(ty, bx)])*(2*confz[cals_coo2D(ty, tx)]-confz[cals_coo2D(by, tx)]-confz[cals_coo2D(by, bx)])
	)+confz[cals_coo2D(ty, tx)] * (
	 (confx[cals_coo2D(ty, tx)]-confx[cals_coo2D(ty, bx)])*(2*confy[cals_coo2D(ty, tx)]-confy[cals_coo2D(by, tx)]-confy[cals_coo2D(by, bx)])
	-(confy[cals_coo2D(ty, tx)]-confy[cals_coo2D(ty, bx)])*(2*confx[cals_coo2D(ty, tx)]-confx[cals_coo2D(by, tx)]-confx[cals_coo2D(by, bx)])
	);
	//(1,0)
	sD[threadIdx.x] += confx[cals_coo2D(typ, tx)] * (
	 (confy[cals_coo2D(typ, tx)]-confy[cals_coo2D(typ, bx)])*(2*confz[cals_coo2D(typ, tx)]-confz[cals_coo2D(ty, tx)]-confz[cals_coo2D(ty, bx)])
	-(confz[cals_coo2D(typ, tx)]-confz[cals_coo2D(typ, bx)])*(2*confy[cals_coo2D(typ, tx)]-confy[cals_coo2D(ty, tx)]-confy[cals_coo2D(ty, bx)])
	)+confy[cals_coo2D(typ, tx)]*(
	 (confz[cals_coo2D(typ, tx)]-confz[cals_coo2D(typ, bx)])*(2*confx[cals_coo2D(typ, tx)]-confx[cals_coo2D(ty, tx)]-confx[cals_coo2D(ty, bx)])
	-(confx[cals_coo2D(typ, tx)]-confx[cals_coo2D(typ, bx)])*(2*confz[cals_coo2D(typ, tx)]-confz[cals_coo2D(ty, tx)]-confz[cals_coo2D(ty, bx)])
	)+confz[cals_coo2D(typ, tx)] * (
	 (confx[cals_coo2D(typ, tx)]-confx[cals_coo2D(typ, bx)])*(2*confy[cals_coo2D(typ, tx)]-confy[cals_coo2D(ty, tx)]-confy[cals_coo2D(ty, bx)])
	-(confy[cals_coo2D(typ, tx)]-confy[cals_coo2D(typ, bx)])*(2*confx[cals_coo2D(typ, tx)]-confx[cals_coo2D(ty, tx)]-confx[cals_coo2D(ty, bx)])
	);
	//(2,0)
	sD[threadIdx.x] += confx[cals_coo2D(typ2, tx)] * (
	 (confy[cals_coo2D(typ2, tx)]-confy[cals_coo2D(typ2, bx)])*(2*confz[cals_coo2D(typ2, tx)]-confz[cals_coo2D(typ, tx)]-confz[cals_coo2D(typ, bx)])
	-(confz[cals_coo2D(typ2, tx)]-confz[cals_coo2D(typ2, bx)])*(2*confy[cals_coo2D(typ2, tx)]-confy[cals_coo2D(typ, tx)]-confy[cals_coo2D(typ, bx)])
	)+confy[cals_coo2D(typ2, tx)]*(
	 (confz[cals_coo2D(typ2, tx)]-confz[cals_coo2D(typ2, bx)])*(2*confx[cals_coo2D(typ2, tx)]-confx[cals_coo2D(typ, tx)]-confx[cals_coo2D(typ, bx)])
	-(confx[cals_coo2D(typ2, tx)]-confx[cals_coo2D(typ2, bx)])*(2*confz[cals_coo2D(typ2, tx)]-confz[cals_coo2D(typ, tx)]-confz[cals_coo2D(typ, bx)])
	)+confz[cals_coo2D(typ2, tx)] * (
	 (confx[cals_coo2D(typ2, tx)]-confx[cals_coo2D(typ2, bx)])*(2*confy[cals_coo2D(typ2, tx)]-confy[cals_coo2D(typ, tx)]-confy[cals_coo2D(typ, bx)])
	-(confy[cals_coo2D(typ2, tx)]-confy[cals_coo2D(typ2, bx)])*(2*confx[cals_coo2D(typ2, tx)]-confx[cals_coo2D(typ, tx)]-confx[cals_coo2D(typ, bx)])
	);
	//(0,1)
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp)] * (
	 (confy[cals_coo2D(ty, txp)]-confy[cals_coo2D(ty, tx)])*(2*confz[cals_coo2D(ty, txp)]-confz[cals_coo2D(by, txp)]-confz[cals_coo2D(by, tx)])
	-(confz[cals_coo2D(ty, txp)]-confz[cals_coo2D(ty, tx)])*(2*confy[cals_coo2D(ty, txp)]-confy[cals_coo2D(by, txp)]-confy[cals_coo2D(by, tx)])
	)+confy[cals_coo2D(ty, txp)]*(
	 (confz[cals_coo2D(ty, txp)]-confz[cals_coo2D(ty, tx)])*(2*confx[cals_coo2D(ty, txp)]-confx[cals_coo2D(by, txp)]-confx[cals_coo2D(by, tx)])
	-(confx[cals_coo2D(ty, txp)]-confx[cals_coo2D(ty, tx)])*(2*confz[cals_coo2D(ty, txp)]-confz[cals_coo2D(by, txp)]-confz[cals_coo2D(by, tx)])
	)+confz[cals_coo2D(ty, txp)] * (
	 (confx[cals_coo2D(ty, txp)]-confx[cals_coo2D(ty, tx)])*(2*confy[cals_coo2D(ty, txp)]-confy[cals_coo2D(by, txp)]-confy[cals_coo2D(by, tx)])
	-(confy[cals_coo2D(ty, txp)]-confy[cals_coo2D(ty, tx)])*(2*confx[cals_coo2D(ty, txp)]-confx[cals_coo2D(by, txp)]-confx[cals_coo2D(by, tx)])
	);
	//(1,1)
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp)] * (
	 (confy[cals_coo2D(typ, txp)]-confy[cals_coo2D(typ, tx)])*(2*confz[cals_coo2D(typ, txp)]-confz[cals_coo2D(ty, txp)]-confz[cals_coo2D(ty, tx)])
	-(confz[cals_coo2D(typ, txp)]-confz[cals_coo2D(typ, tx)])*(2*confy[cals_coo2D(typ, txp)]-confy[cals_coo2D(ty, txp)]-confy[cals_coo2D(ty, tx)])
	)+confy[cals_coo2D(typ, txp)]*(
	 (confz[cals_coo2D(typ, txp)]-confz[cals_coo2D(typ, tx)])*(2*confx[cals_coo2D(typ, txp)]-confx[cals_coo2D(ty, txp)]-confx[cals_coo2D(ty, tx)])
	-(confx[cals_coo2D(typ, txp)]-confx[cals_coo2D(typ, tx)])*(2*confz[cals_coo2D(typ, txp)]-confz[cals_coo2D(ty, txp)]-confz[cals_coo2D(ty, tx)])
	)+confz[cals_coo2D(typ, txp)] * (
	 (confx[cals_coo2D(typ, txp)]-confx[cals_coo2D(typ, tx)])*(2*confy[cals_coo2D(typ, txp)]-confy[cals_coo2D(ty, txp)]-confy[cals_coo2D(ty, tx)])
	-(confy[cals_coo2D(typ, txp)]-confy[cals_coo2D(typ, tx)])*(2*confx[cals_coo2D(typ, txp)]-confx[cals_coo2D(ty, txp)]-confx[cals_coo2D(ty, tx)])
	);
	//(2,1)
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp)] * (
	 (confy[cals_coo2D(typ2, txp)]-confy[cals_coo2D(typ2, tx)])*(2*confz[cals_coo2D(typ2, txp)]-confz[cals_coo2D(typ, txp)]-confz[cals_coo2D(typ, tx)])
	-(confz[cals_coo2D(typ2, txp)]-confz[cals_coo2D(typ2, tx)])*(2*confy[cals_coo2D(typ2, txp)]-confy[cals_coo2D(typ, txp)]-confy[cals_coo2D(typ, tx)])
	)+confy[cals_coo2D(typ2, txp)]*(
	 (confz[cals_coo2D(typ2, txp)]-confz[cals_coo2D(typ2, tx)])*(2*confx[cals_coo2D(typ2, txp)]-confx[cals_coo2D(typ, txp)]-confx[cals_coo2D(typ, tx)])
	-(confx[cals_coo2D(typ2, txp)]-confx[cals_coo2D(typ2, tx)])*(2*confz[cals_coo2D(typ2, txp)]-confz[cals_coo2D(typ, txp)]-confz[cals_coo2D(typ, tx)])
	)+confz[cals_coo2D(typ2, txp)] * (
	 (confx[cals_coo2D(typ2, txp)]-confx[cals_coo2D(typ2, tx)])*(2*confy[cals_coo2D(typ2, txp)]-confy[cals_coo2D(typ, txp)]-confy[cals_coo2D(typ, tx)])
	-(confy[cals_coo2D(typ2, txp)]-confy[cals_coo2D(typ2, tx)])*(2*confx[cals_coo2D(typ2, txp)]-confx[cals_coo2D(typ, txp)]-confx[cals_coo2D(typ, tx)])
	);
	//(0,2)
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp2)] * (
	 (confy[cals_coo2D(ty, txp2)]-confy[cals_coo2D(ty, txp)])*(2*confz[cals_coo2D(ty, txp2)]-confz[cals_coo2D(by, txp2)]-confz[cals_coo2D(by, txp)])
	-(confz[cals_coo2D(ty, txp2)]-confz[cals_coo2D(ty, txp)])*(2*confy[cals_coo2D(ty, txp2)]-confy[cals_coo2D(by, txp2)]-confy[cals_coo2D(by, txp)])
	)+confy[cals_coo2D(ty, txp2)]*(
	 (confz[cals_coo2D(ty, txp2)]-confz[cals_coo2D(ty, txp)])*(2*confx[cals_coo2D(ty, txp2)]-confx[cals_coo2D(by, txp2)]-confx[cals_coo2D(by, txp)])
	-(confx[cals_coo2D(ty, txp2)]-confx[cals_coo2D(ty, txp)])*(2*confz[cals_coo2D(ty, txp2)]-confz[cals_coo2D(by, txp2)]-confz[cals_coo2D(by, txp)])
	)+confz[cals_coo2D(ty, txp2)] * (
	 (confx[cals_coo2D(ty, txp2)]-confx[cals_coo2D(ty, txp)])*(2*confy[cals_coo2D(ty, txp2)]-confy[cals_coo2D(by, txp2)]-confy[cals_coo2D(by, txp)])
	-(confy[cals_coo2D(ty, txp2)]-confy[cals_coo2D(ty, txp)])*(2*confx[cals_coo2D(ty, txp2)]-confx[cals_coo2D(by, txp2)]-confx[cals_coo2D(by, txp)])
	);
	//(1,2)
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp2)] * (
	 (confy[cals_coo2D(typ, txp2)]-confy[cals_coo2D(typ, txp)])*(2*confz[cals_coo2D(typ, txp2)]-confz[cals_coo2D(ty, txp2)]-confz[cals_coo2D(ty, txp)])
	-(confz[cals_coo2D(typ, txp2)]-confz[cals_coo2D(typ, txp)])*(2*confy[cals_coo2D(typ, txp2)]-confy[cals_coo2D(ty, txp2)]-confy[cals_coo2D(ty, txp)])
	)+confy[cals_coo2D(typ, txp2)]*(
	 (confz[cals_coo2D(typ, txp2)]-confz[cals_coo2D(typ, txp)])*(2*confx[cals_coo2D(typ, txp2)]-confx[cals_coo2D(ty, txp2)]-confx[cals_coo2D(ty, txp)])
	-(confx[cals_coo2D(typ, txp2)]-confx[cals_coo2D(typ, txp)])*(2*confz[cals_coo2D(typ, txp2)]-confz[cals_coo2D(ty, txp2)]-confz[cals_coo2D(ty, txp)])
	)+confz[cals_coo2D(typ, txp2)] * (
	 (confx[cals_coo2D(typ, txp2)]-confx[cals_coo2D(typ, txp)])*(2*confy[cals_coo2D(typ, txp2)]-confy[cals_coo2D(ty, txp2)]-confy[cals_coo2D(ty, txp)])
	-(confy[cals_coo2D(typ, txp2)]-confy[cals_coo2D(typ, txp)])*(2*confx[cals_coo2D(typ, txp2)]-confx[cals_coo2D(ty, txp2)]-confx[cals_coo2D(ty, txp)])
	);
	//(2,2)
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp2)] * (
	 (confy[cals_coo2D(typ2, txp2)]-confy[cals_coo2D(typ2, txp)])*(2*confz[cals_coo2D(typ2, txp2)]-confz[cals_coo2D(typ, txp2)]-confz[cals_coo2D(typ, txp)])
	-(confz[cals_coo2D(typ2, txp2)]-confz[cals_coo2D(typ2, txp)])*(2*confy[cals_coo2D(typ2, txp2)]-confy[cals_coo2D(typ, txp2)]-confy[cals_coo2D(typ, txp)])
	)+confy[cals_coo2D(typ2, txp2)]*(
	 (confz[cals_coo2D(typ2, txp2)]-confz[cals_coo2D(typ2, txp)])*(2*confx[cals_coo2D(typ2, txp2)]-confx[cals_coo2D(typ, txp2)]-confx[cals_coo2D(typ, txp)])
	-(confx[cals_coo2D(typ2, txp2)]-confx[cals_coo2D(typ2, txp)])*(2*confz[cals_coo2D(typ2, txp2)]-confz[cals_coo2D(typ, txp2)]-confz[cals_coo2D(typ, txp)])
	)+confz[cals_coo2D(typ2, txp2)] * (
	 (confx[cals_coo2D(typ2, txp2)]-confx[cals_coo2D(typ2, txp)])*(2*confy[cals_coo2D(typ2, txp2)]-confy[cals_coo2D(typ, txp2)]-confy[cals_coo2D(typ, txp)])
	-(confy[cals_coo2D(typ2, txp2)]-confy[cals_coo2D(typ2, txp)])*(2*confx[cals_coo2D(typ2, txp2)]-confx[cals_coo2D(typ, txp2)]-confx[cals_coo2D(typ, txp)])
	);
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>=512){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>=256){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>=128){
		if((threadIdx.x < 64) && (threadIdx.x+64 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+64];
		}
		__syncthreads();
	}
	if(threadIdx.x < 32){
		sD[threadIdx.x] += sD[threadIdx.x+32];
	}
	__syncthreads();
	if(threadIdx.x < 16){
		sD[threadIdx.x] += sD[threadIdx.x+16];
	}
	__syncthreads();
	if(threadIdx.x < 8){
		sD[threadIdx.x] += sD[threadIdx.x+8];
	}
	__syncthreads();
	if(threadIdx.x < 4){
		sD[threadIdx.x] += sD[threadIdx.x+4];
	}
	__syncthreads();
	if(threadIdx.x < 2){
		sD[threadIdx.x] += sD[threadIdx.x+2];
	}
	__syncthreads();
	if(threadIdx.x < 1){
		sD[threadIdx.x] += sD[threadIdx.x+1];
	}
	__syncthreads();
	if(threadIdx.x == 0)
		out[dataoff + (blockIdx.x % cals_BN) + 4*cals_BN] = sD[0];
	__syncthreads();
}
#endif
