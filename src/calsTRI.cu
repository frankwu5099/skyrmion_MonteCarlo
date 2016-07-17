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
__constant__ float Q1x;
__constant__ float Q1y;
__constant__ float Q2x;
__constant__ float Q2y;
__constant__ float cals_A; //(0.0)
__constant__ float cBXPyz;
__constant__ float cBYPyz;
__constant__ float cBWPyz;
__constant__ float cBXMyz;
__constant__ float cBYMyz;
__constant__ float cBWMyz;
__constant__ float cBXPzy;
__constant__ float cBYPzy;
__constant__ float cBWPzy;
__constant__ float cBXMzy;
__constant__ float cBYMzy;
__constant__ float cBWMzy;
__constant__ float cBXPxz;
__constant__ float cBYPxz;
__constant__ float cBWPxz;
__constant__ float cBXMxz;
__constant__ float cBYMxz;
__constant__ float cBWMxz;
__constant__ float cBXPzx;
__constant__ float cBYPzx;
__constant__ float cBWPzx;
__constant__ float cBXMzx;
__constant__ float cBYMzx;
__constant__ float cBWMzx;
void move_params_device_cals(){
  float tmpp;
  cudaMemcpyToSymbol( cals_SpinSize, &H_SpinSize, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_SpinSize_z, &H_SpinSize_z, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BlockSize_x, &H_BlockSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BlockSize_y, &H_BlockSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_GridSize_x, &H_GridSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_GridSize_y, &H_GridSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_TN, &H_TN, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BN, &H_BN, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_A , &H_A , sizeof(float));
  cudaMemcpyToSymbol( Q1x , &H_Q1x , sizeof(float));
  cudaMemcpyToSymbol( Q1y , &H_Q1y , sizeof(float));
  cudaMemcpyToSymbol( Q2x , &H_Q2x , sizeof(float));
  cudaMemcpyToSymbol( Q2y , &H_Q2y , sizeof(float));
  tmpp = (DD);
  cudaMemcpyToSymbol( cBXPyz, &tmpp, sizeof(float));
  tmpp = (-0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBYPyz, &tmpp, sizeof(float));
  tmpp = (0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBWPyz, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( cBXMyz, &tmpp, sizeof(float));
  tmpp = (0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBYMyz, &tmpp, sizeof(float));
  tmpp = (-0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBWMyz, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( cBXPzy, &tmpp, sizeof(float));
  tmpp =  (0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBYPzy, &tmpp, sizeof(float));
  tmpp =  (-0.5 * DD - sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBWPzy, &tmpp, sizeof(float));
  tmpp = (DD);
  cudaMemcpyToSymbol( cBXMzy, &tmpp, sizeof(float));
  tmpp = (-0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBYMzy, &tmpp, sizeof(float));
  tmpp = (0.5 * DD + sqrt3d2 * DR);
  cudaMemcpyToSymbol( cBWMzy, &tmpp, sizeof(float));
  tmpp = (DR);
  cudaMemcpyToSymbol( cBXPxz, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( cBYPxz, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( cBWPxz, &tmpp, sizeof(float));
  tmpp = (-DR);
  cudaMemcpyToSymbol( cBXMxz, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( cBYMxz, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( cBWMxz, &tmpp, sizeof(float));
  tmpp = (-DR);
  cudaMemcpyToSymbol( cBXPzx, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( cBYPzx, &tmpp, sizeof(float));
  tmpp = (-sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( cBWPzx, &tmpp, sizeof(float));
  tmpp = (DR);
  cudaMemcpyToSymbol( cBXMzx, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD - 0.5 * DR);
  cudaMemcpyToSymbol( cBYMzx, &tmpp, sizeof(float));
  tmpp = (sqrt3d2 * DD + 0.5 * DR);
  cudaMemcpyToSymbol( cBWMzx, &tmpp, sizeof(float));
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
	int bx, by, tx_ty = tx + (ty % cals_SpinSize);
	//-----Calculate the energy of each spin pairs in the system-----
	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.

	bx = (tx + cals_SpinSize - 1) % cals_SpinSize;
	if((ty % cals_SpinSize) == 0)	by = ty + cals_SpinSize - 1;
	else				by = ty - 1;
	//Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.

	//0,0
	sD[threadIdx.x] = -confx[cals_coo2D(ty, tx)] * ( BXMxx * confx[cals_coo2D(ty, bx)] + BYMxx * confx[cals_coo2D(by, tx)] + BWMxx * confx[cals_coo2D(by, bx)])\
	           -confx[cals_coo2D(ty, tx)] * ( BXMxy * confy[cals_coo2D(ty, bx)] + BYMxy * confy[cals_coo2D(by, tx)] + BWMxy * confy[cals_coo2D(by, bx)])\
	           -confx[cals_coo2D(ty, tx)] * ( cBXMxz * confz[cals_coo2D(ty, bx)] + cBYMxz * confz[cals_coo2D(by, tx)] + cBWMxz * confz[cals_coo2D(by, bx)])\
		         -confy[cals_coo2D(ty, tx)] * ( BXMyx * confx[cals_coo2D(ty, bx)] + BYMyx * confx[cals_coo2D(by, tx)] + BWMyx * confx[cals_coo2D(by, bx)])\
		         -confy[cals_coo2D(ty, tx)] * ( BXMyy * confy[cals_coo2D(ty, bx)] + BYMyy * confy[cals_coo2D(by, tx)] + BWMyy * confy[cals_coo2D(by, bx)])\
		         -confy[cals_coo2D(ty, tx)] * ( cBXMyz * confz[cals_coo2D(ty, bx)] + cBYMyz * confz[cals_coo2D(by, tx)] + cBWMyz * confz[cals_coo2D(by, bx)])\
		         -confz[cals_coo2D(ty, tx)] * ( cBXMzx * confx[cals_coo2D(ty, bx)] + cBYMzx * confx[cals_coo2D(by, tx)] + cBWMzx * confx[cals_coo2D(by, bx)])\
		         -confz[cals_coo2D(ty, tx)] * ( cBXMzy * confy[cals_coo2D(ty, bx)] + cBYMzy * confy[cals_coo2D(by, tx)] + cBWMzy * confy[cals_coo2D(by, bx)])\
		         -confz[cals_coo2D(ty, tx)] * ( BXMzz * confz[cals_coo2D(ty, bx)] + BYMzz * confz[cals_coo2D(by, tx)] + BWMzz * confz[cals_coo2D(by, bx)] - cals_A * confz[cals_coo2D(ty, tx)]);
	//1,0
	sD[threadIdx.x] -= confx[cals_coo2D(typ, tx)] * ( BXMxx * confx[cals_coo2D(typ, bx)] + BYMxx * confx[cals_coo2D(ty, tx)] + BWMxx * confx[cals_coo2D(ty, bx)])\
		         +confx[cals_coo2D(typ, tx)] * ( BXMxy * confy[cals_coo2D(typ, bx)] + BYMxy * confy[cals_coo2D(ty, tx)] + BWMxy * confy[cals_coo2D(ty, bx)])\
		         +confx[cals_coo2D(typ, tx)] * ( cBXMxz * confz[cals_coo2D(typ, bx)] + cBYMxz * confz[cals_coo2D(ty, tx)] + cBWMxz * confz[cals_coo2D(ty, bx)])\
		         +confy[cals_coo2D(typ, tx)] * ( BXMyx * confx[cals_coo2D(typ, bx)] + BYMyx * confx[cals_coo2D(ty, tx)] + BWMyx * confx[cals_coo2D(ty, bx)])\
		         +confy[cals_coo2D(typ, tx)] * ( BXMyy * confy[cals_coo2D(typ, bx)] + BYMyy * confy[cals_coo2D(ty, tx)] + BWMyy * confy[cals_coo2D(ty, bx)])\
		         +confy[cals_coo2D(typ, tx)] * ( cBXMyz * confz[cals_coo2D(typ, bx)] + cBYMyz * confz[cals_coo2D(ty, tx)] + cBWMyz * confz[cals_coo2D(ty, bx)])\
		         +confz[cals_coo2D(typ, tx)] * ( cBXMzx * confx[cals_coo2D(typ, bx)] + cBYMzx * confx[cals_coo2D(ty, tx)] + cBWMzx * confx[cals_coo2D(ty, bx)])\
		         +confz[cals_coo2D(typ, tx)] * ( cBXMzy * confy[cals_coo2D(typ, bx)] + cBYMzy * confy[cals_coo2D(ty, tx)] + cBWMzy * confy[cals_coo2D(ty, bx)])\
		         +confz[cals_coo2D(typ, tx)] * ( BXMzz * confz[cals_coo2D(typ, bx)] + BYMzz * confz[cals_coo2D(ty, tx)] + BWMzz * confz[cals_coo2D(ty, bx)] - cals_A * confz[cals_coo2D((ty+1), tx)]);
	//2,0
	sD[threadIdx.x] -= confx[cals_coo2D(typ2, tx)] * ( BXMxx * confx[cals_coo2D(typ2, bx)] + BYMxx * confx[cals_coo2D(typ, tx)] + BWMxx * confx[cals_coo2D(typ, bx)])\
		         +confx[cals_coo2D(typ2, tx)] * ( BXMxy * confy[cals_coo2D(typ2, bx)] + BYMxy * confy[cals_coo2D(typ, tx)] + BWMxy * confy[cals_coo2D(typ, bx)])\
		         +confx[cals_coo2D(typ2, tx)] * ( cBXMxz * confz[cals_coo2D(typ2, bx)] + cBYMxz * confz[cals_coo2D(typ, tx)] + cBWMxz * confz[cals_coo2D(typ, bx)])\
		         +confy[cals_coo2D(typ2, tx)] * ( BXMyx * confx[cals_coo2D(typ2, bx)] + BYMyx * confx[cals_coo2D(typ, tx)] + BWMyx * confx[cals_coo2D(typ, bx)])\
		         +confy[cals_coo2D(typ2, tx)] * ( BXMyy * confy[cals_coo2D(typ2, bx)] + BYMyy * confy[cals_coo2D(typ, tx)] + BWMyy * confy[cals_coo2D(typ, bx)])\
		         +confy[cals_coo2D(typ2, tx)] * ( cBXMyz * confz[cals_coo2D(typ2, bx)] + cBYMyz * confz[cals_coo2D(typ, tx)] + cBWMyz * confz[cals_coo2D(typ, bx)])\
		         +confz[cals_coo2D(typ2, tx)] * ( cBXMzx * confx[cals_coo2D(typ2, bx)] + cBYMzx * confx[cals_coo2D(typ, tx)] + cBWMzx * confx[cals_coo2D(typ, bx)])\
		         +confz[cals_coo2D(typ2, tx)] * ( cBXMzy * confy[cals_coo2D(typ2, bx)] + cBYMzy * confy[cals_coo2D(typ, tx)] + cBWMzy * confy[cals_coo2D(typ, bx)])\
		         +confz[cals_coo2D(typ2, tx)] * ( BXMzz * confz[cals_coo2D(typ2, bx)] + BYMzz * confz[cals_coo2D(typ, tx)] + BWMzz * confz[cals_coo2D(typ, bx)] - cals_A * confz[cals_coo2D((ty+1), tx)]);
	//0,1
	sD[threadIdx.x] -= confx[cals_coo2D(ty, txp)] * ( BXMxx * confx[cals_coo2D(ty, tx)] + BYMxx * confx[cals_coo2D(by, txp)] + BWMxx * confx[cals_coo2D(by, tx)])\
		         +confx[cals_coo2D(ty, txp)] * ( BXMxy * confy[cals_coo2D(ty, tx)] + BYMxy * confy[cals_coo2D(by, txp)] + BWMxy * confy[cals_coo2D(by, tx)])\
		         +confx[cals_coo2D(ty, txp)] * ( cBXMxz * confz[cals_coo2D(ty, tx)] + cBYMxz * confz[cals_coo2D(by, txp)] + cBWMxz * confz[cals_coo2D(by, tx)])\
		         +confy[cals_coo2D(ty, txp)] * ( BXMyx * confx[cals_coo2D(ty, tx)] + BYMyx * confx[cals_coo2D(by, txp)] + BWMyx * confx[cals_coo2D(by, tx)])\
		         +confy[cals_coo2D(ty, txp)] * ( BXMyy * confy[cals_coo2D(ty, tx)] + BYMyy * confy[cals_coo2D(by, txp)] + BWMyy * confy[cals_coo2D(by, tx)])\
		         +confy[cals_coo2D(ty, txp)] * ( cBXMyz * confz[cals_coo2D(ty, tx)] + cBYMyz * confz[cals_coo2D(by, txp)] + cBWMyz * confz[cals_coo2D(by, tx)])\
		         +confz[cals_coo2D(ty, txp)] * ( cBXMzx * confx[cals_coo2D(ty, tx)] + cBYMzx * confx[cals_coo2D(by, txp)] + cBWMzx * confx[cals_coo2D(by, tx)])\
		         +confz[cals_coo2D(ty, txp)] * ( cBXMzy * confy[cals_coo2D(ty, tx)] + cBYMzy * confy[cals_coo2D(by, txp)] + cBWMzy * confy[cals_coo2D(by, tx)])\
		         +confz[cals_coo2D(ty, txp)] * ( BXMzz * confz[cals_coo2D(ty, tx)] + BYMzz * confz[cals_coo2D(by, txp)] + BWMzz * confz[cals_coo2D(by, tx)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//1,1
	sD[threadIdx.x] -= confx[cals_coo2D(typ, txp)] * ( BXMxx * confx[cals_coo2D(typ, tx)] + BYMxx * confx[cals_coo2D(ty, txp)] + BWMxx * confx[cals_coo2D(ty, tx)])\
		         +confx[cals_coo2D(typ, txp)] * ( BXMxy * confy[cals_coo2D(typ, tx)] + BYMxy * confy[cals_coo2D(ty, txp)] + BWMxy * confy[cals_coo2D(ty, tx)])\
		         +confx[cals_coo2D(typ, txp)] * ( cBXMxz * confz[cals_coo2D(typ, tx)] + cBYMxz * confz[cals_coo2D(ty, txp)] + cBWMxz * confz[cals_coo2D(ty, tx)])\
		         +confy[cals_coo2D(typ, txp)] * ( BXMyx * confx[cals_coo2D(typ, tx)] + BYMyx * confx[cals_coo2D(ty, txp)] + BWMyx * confx[cals_coo2D(ty, tx)])\
		         +confy[cals_coo2D(typ, txp)] * ( BXMyy * confy[cals_coo2D(typ, tx)] + BYMyy * confy[cals_coo2D(ty, txp)] + BWMyy * confy[cals_coo2D(ty, tx)])\
		         +confy[cals_coo2D(typ, txp)] * ( cBXMyz * confz[cals_coo2D(typ, tx)] + cBYMyz * confz[cals_coo2D(ty, txp)] + cBWMyz * confz[cals_coo2D(ty, tx)])\
		         +confz[cals_coo2D(typ, txp)] * ( cBXMzx * confx[cals_coo2D(typ, tx)] + cBYMzx * confx[cals_coo2D(ty, txp)] + cBWMzx * confx[cals_coo2D(ty, tx)])\
		         +confz[cals_coo2D(typ, txp)] * ( cBXMzy * confy[cals_coo2D(typ, tx)] + cBYMzy * confy[cals_coo2D(ty, txp)] + cBWMzy * confy[cals_coo2D(ty, tx)])\
		         +confz[cals_coo2D(typ, txp)] * ( BXMzz * confz[cals_coo2D(typ, tx)] + BYMzz * confz[cals_coo2D(ty, txp)] + BWMzz * confz[cals_coo2D(ty, tx)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//2,1
	sD[threadIdx.x] -= confx[cals_coo2D(typ2, txp)] * ( BXMxx * confx[cals_coo2D(typ2, tx)] + BYMxx * confx[cals_coo2D(typ, txp)] + BWMxx * confx[cals_coo2D(typ, tx)])\
		         +confx[cals_coo2D(typ2, txp)] * ( BXMxy * confy[cals_coo2D(typ2, tx)] + BYMxy * confy[cals_coo2D(typ, txp)] + BWMxy * confy[cals_coo2D(typ, tx)])\
		         +confx[cals_coo2D(typ2, txp)] * ( cBXMxz * confz[cals_coo2D(typ2, tx)] + cBYMxz * confz[cals_coo2D(typ, txp)] + cBWMxz * confz[cals_coo2D(typ, tx)])\
		         +confy[cals_coo2D(typ2, txp)] * ( BXMyx * confx[cals_coo2D(typ2, tx)] + BYMyx * confx[cals_coo2D(typ, txp)] + BWMyx * confx[cals_coo2D(typ, tx)])\
		         +confy[cals_coo2D(typ2, txp)] * ( BXMyy * confy[cals_coo2D(typ2, tx)] + BYMyy * confy[cals_coo2D(typ, txp)] + BWMyy * confy[cals_coo2D(typ, tx)])\
		         +confy[cals_coo2D(typ2, txp)] * ( cBXMyz * confz[cals_coo2D(typ2, tx)] + cBYMyz * confz[cals_coo2D(typ, txp)] + cBWMyz * confz[cals_coo2D(typ, tx)])\
		         +confz[cals_coo2D(typ2, txp)] * ( cBXMzx * confx[cals_coo2D(typ2, tx)] + cBYMzx * confx[cals_coo2D(typ, txp)] + cBWMzx * confx[cals_coo2D(typ, tx)])\
		         +confz[cals_coo2D(typ2, txp)] * ( cBXMzy * confy[cals_coo2D(typ2, tx)] + cBYMzy * confy[cals_coo2D(typ, txp)] + cBWMzy * confy[cals_coo2D(typ, tx)])\
		         +confz[cals_coo2D(typ2, txp)] * ( BXMzz * confz[cals_coo2D(typ2, tx)] + BYMzz * confz[cals_coo2D(typ, txp)] + BWMzz * confz[cals_coo2D(typ, tx)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//0,2
	sD[threadIdx.x] -= confx[cals_coo2D(ty, txp2)] * ( BXMxx * confx[cals_coo2D(ty, txp)] + BYMxx * confx[cals_coo2D(by, txp2)] + BWMxx * confx[cals_coo2D(by, txp)])\
		         +confx[cals_coo2D(ty, txp2)] * ( BXMxy * confy[cals_coo2D(ty, txp)] + BYMxy * confy[cals_coo2D(by, txp2)] + BWMxy * confy[cals_coo2D(by, txp)])\
		         +confx[cals_coo2D(ty, txp2)] * ( cBXMxz * confz[cals_coo2D(ty, txp)] + cBYMxz * confz[cals_coo2D(by, txp2)] + cBWMxz * confz[cals_coo2D(by, txp)])\
		         +confy[cals_coo2D(ty, txp2)] * ( BXMyx * confx[cals_coo2D(ty, txp)] + BYMyx * confx[cals_coo2D(by, txp2)] + BWMyx * confx[cals_coo2D(by, txp)])\
		         +confy[cals_coo2D(ty, txp2)] * ( BXMyy * confy[cals_coo2D(ty, txp)] + BYMyy * confy[cals_coo2D(by, txp2)] + BWMyy * confy[cals_coo2D(by, txp)])\
		         +confy[cals_coo2D(ty, txp2)] * ( cBXMyz * confz[cals_coo2D(ty, txp)] + cBYMyz * confz[cals_coo2D(by, txp2)] + cBWMyz * confz[cals_coo2D(by, txp)])\
		         +confz[cals_coo2D(ty, txp2)] * ( cBXMzx * confx[cals_coo2D(ty, txp)] + cBYMzx * confx[cals_coo2D(by, txp2)] + cBWMzx * confx[cals_coo2D(by, txp)])\
		         +confz[cals_coo2D(ty, txp2)] * ( cBXMzy * confy[cals_coo2D(ty, txp)] + cBYMzy * confy[cals_coo2D(by, txp2)] + cBWMzy * confy[cals_coo2D(by, txp)])\
		         +confz[cals_coo2D(ty, txp2)] * ( BXMzz * confz[cals_coo2D(ty, txp)] + BYMzz * confz[cals_coo2D(by, txp2)] + BWMzz * confz[cals_coo2D(by, txp)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//1,2
	sD[threadIdx.x] -= confx[cals_coo2D(typ, txp2)] * ( BXMxx * confx[cals_coo2D(typ, txp)] + BYMxx * confx[cals_coo2D(ty, txp2)] + BWMxx * confx[cals_coo2D(ty, txp)])\
		         +confx[cals_coo2D(typ, txp2)] * ( BXMxy * confy[cals_coo2D(typ, txp)] + BYMxy * confy[cals_coo2D(ty, txp2)] + BWMxy * confy[cals_coo2D(ty, txp)])\
		         +confx[cals_coo2D(typ, txp2)] * ( cBXMxz * confz[cals_coo2D(typ, txp)] + cBYMxz * confz[cals_coo2D(ty, txp2)] + cBWMxz * confz[cals_coo2D(ty, txp)])\
		         +confy[cals_coo2D(typ, txp2)] * ( BXMyx * confx[cals_coo2D(typ, txp)] + BYMyx * confx[cals_coo2D(ty, txp2)] + BWMyx * confx[cals_coo2D(ty, txp)])\
		         +confy[cals_coo2D(typ, txp2)] * ( BXMyy * confy[cals_coo2D(typ, txp)] + BYMyy * confy[cals_coo2D(ty, txp2)] + BWMyy * confy[cals_coo2D(ty, txp)])\
		         +confy[cals_coo2D(typ, txp2)] * ( cBXMyz * confz[cals_coo2D(typ, txp)] + cBYMyz * confz[cals_coo2D(ty, txp2)] + cBWMyz * confz[cals_coo2D(ty, txp)])\
		         +confz[cals_coo2D(typ, txp2)] * ( cBXMzx * confx[cals_coo2D(typ, txp)] + cBYMzx * confx[cals_coo2D(ty, txp2)] + cBWMzx * confx[cals_coo2D(ty, txp)])\
		         +confz[cals_coo2D(typ, txp2)] * ( cBXMzy * confy[cals_coo2D(typ, txp)] + cBYMzy * confy[cals_coo2D(ty, txp2)] + cBWMzy * confy[cals_coo2D(ty, txp)])\
		         +confz[cals_coo2D(typ, txp2)] * ( BXMzz * confz[cals_coo2D(typ, txp)] + BYMzz * confz[cals_coo2D(ty, txp2)] + BWMzz * confz[cals_coo2D(ty, txp)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	//2,2
	sD[threadIdx.x] -= confx[cals_coo2D(typ2, txp2)] * ( BXMxx * confx[cals_coo2D(typ2, txp)] + BYMxx * confx[cals_coo2D(typ, txp2)] + BWMxx * confx[cals_coo2D(typ, txp)])\
		         +confx[cals_coo2D(typ2, txp2)] * ( BXMxy * confy[cals_coo2D(typ2, txp)] + BYMxy * confy[cals_coo2D(typ, txp2)] + BWMxy * confy[cals_coo2D(typ, txp)])\
		         +confx[cals_coo2D(typ2, txp2)] * ( cBXMxz * confz[cals_coo2D(typ2, txp)] + cBYMxz * confz[cals_coo2D(typ, txp2)] + cBWMxz * confz[cals_coo2D(typ, txp)])\
		         +confy[cals_coo2D(typ2, txp2)] * ( BXMyx * confx[cals_coo2D(typ2, txp)] + BYMyx * confx[cals_coo2D(typ, txp2)] + BWMyx * confx[cals_coo2D(typ, txp)])\
		         +confy[cals_coo2D(typ2, txp2)] * ( BXMyy * confy[cals_coo2D(typ2, txp)] + BYMyy * confy[cals_coo2D(typ, txp2)] + BWMyy * confy[cals_coo2D(typ, txp)])\
		         +confy[cals_coo2D(typ2, txp2)] * ( cBXMyz * confz[cals_coo2D(typ2, txp)] + cBYMyz * confz[cals_coo2D(typ, txp2)] + cBWMyz * confz[cals_coo2D(typ, txp)])\
		         +confz[cals_coo2D(typ2, txp2)] * ( cBXMzx * confx[cals_coo2D(typ2, txp)] + cBYMzx * confx[cals_coo2D(typ, txp2)] + cBWMzx * confx[cals_coo2D(typ, txp)])\
		         +confz[cals_coo2D(typ2, txp2)] * ( cBXMzy * confy[cals_coo2D(typ2, txp)] + cBYMzy * confy[cals_coo2D(typ, txp2)] + cBWMzy * confy[cals_coo2D(typ, txp)])\
		         +confz[cals_coo2D(typ2, txp2)] * ( BXMzz * confz[cals_coo2D(typ2, txp)] + BYMzz * confz[cals_coo2D(typ, txp2)] + BWMzz * confz[cals_coo2D(typ, txp)] - cals_A * confz[cals_coo2D(ty, tx+1)]);
	__syncthreads();


	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
	sD[threadIdx.x]  = confz[cals_coo2D(ty, tx)];
	sD[threadIdx.x] += confz[cals_coo2D(typ, tx)];
	sD[threadIdx.x] += confz[cals_coo2D(typ2, tx)];
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp)];
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp)];
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp)];
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp2)];
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp2)];
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp2)];
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confx[cals_coo2D(ty, tx)]     * cosf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, tx)]    * cosf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, tx)]   * cosf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp)]    * cosf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp)]   * cosf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp)]  * cosf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp2)]   * cosf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp2)]  * cosf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp2)] * cosf(Q1x*(txp2) + Q1y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 5*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confy[cals_coo2D(ty, tx)]     * cosf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, tx)]    * cosf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, tx)]   * cosf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp)]    * cosf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp)]   * cosf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp)]  * cosf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp2)]   * cosf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp2)]  * cosf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp2)] * cosf(Q1x*(txp2) + Q1y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 6*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confz[cals_coo2D(ty, tx)]     * cosf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, tx)]    * cosf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, tx)]   * cosf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp)]    * cosf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp)]   * cosf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp)]  * cosf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp2)]   * cosf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp2)]  * cosf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp2)] * cosf(Q1x*(txp2) + Q1y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 7*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confx[cals_coo2D(ty, tx)]     * sinf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, tx)]    * sinf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, tx)]   * sinf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp)]    * sinf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp)]   * sinf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp)]  * sinf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp2)]   * sinf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp2)]  * sinf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp2)] * sinf(Q1x*(txp2) + Q1y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 8*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confy[cals_coo2D(ty, tx)]     * sinf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, tx)]    * sinf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, tx)]   * sinf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp)]    * sinf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp)]   * sinf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp)]  * sinf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp2)]   * sinf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp2)]  * sinf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp2)] * sinf(Q1x*(txp2) + Q1y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 9*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confz[cals_coo2D(ty, tx)]     * sinf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, tx)]    * sinf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, tx)]   * sinf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp)]    * sinf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp)]   * sinf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp)]  * sinf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp2)]   * sinf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp2)]  * sinf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp2)] * sinf(Q1x*(txp2) + Q1y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 10*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confx[cals_coo2D(ty, tx)]     * cosf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, tx)]    * cosf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, tx)]   * cosf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp)]    * cosf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp)]   * cosf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp)]  * cosf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp2)]   * cosf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp2)]  * cosf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp2)] * cosf(Q2x*(txp2) + Q2y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 11*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confy[cals_coo2D(ty, tx)]     * cosf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, tx)]    * cosf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, tx)]   * cosf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp)]    * cosf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp)]   * cosf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp)]  * cosf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp2)]   * cosf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp2)]  * cosf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp2)] * cosf(Q2x*(txp2) + Q2y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 12*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confz[cals_coo2D(ty, tx)]     * cosf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, tx)]    * cosf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, tx)]   * cosf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp)]    * cosf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp)]   * cosf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp)]  * cosf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp2)]   * cosf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp2)]  * cosf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp2)] * cosf(Q2x*(txp2) + Q2y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 13*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confx[cals_coo2D(ty, tx)]     * sinf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, tx)]    * sinf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, tx)]   * sinf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp)]    * sinf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp)]   * sinf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp)]  * sinf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo2D(ty, txp2)]   * sinf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo2D(typ, txp2)]  * sinf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo2D(typ2, txp2)] * sinf(Q2x*(txp2) + Q2y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 14*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confy[cals_coo2D(ty, tx)]     * sinf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, tx)]    * sinf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, tx)]   * sinf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp)]    * sinf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp)]   * sinf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp)]  * sinf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo2D(ty, txp2)]   * sinf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo2D(typ, txp2)]  * sinf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo2D(typ2, txp2)] * sinf(Q2x*(txp2) + Q2y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 15*cals_BN] = sD[0];
	__syncthreads();

	//Sum over the magnetic moments in x direction of the eight spins on each thread cubic and store the result of each thread cubic in sD.
	sD[threadIdx.x]  = confz[cals_coo2D(ty, tx)]     * sinf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, tx)]    * sinf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, tx)]   * sinf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp)]    * sinf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp)]   * sinf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp)]  * sinf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo2D(ty, txp2)]   * sinf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo2D(typ, txp2)]  * sinf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo2D(typ2, txp2)] * sinf(Q2x*(txp2) + Q2y*(typ2));
	__syncthreads();

	//Sum over all elements in each sD
	if(cals_TN>256){
		if((threadIdx.x < 256) && (threadIdx.x+256 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+256];
		}
		__syncthreads();
	}
	if(cals_TN>128){
		if((threadIdx.x < 128) && (threadIdx.x+128 < cals_TN)){
			sD[threadIdx.x] += sD[threadIdx.x+128];
		}
		__syncthreads();
	}
	if(cals_TN>64){
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
		out[dataoff + (blockIdx.x % cals_BN) + 16*cals_BN] = sD[0];
	__syncthreads();
}
#endif
