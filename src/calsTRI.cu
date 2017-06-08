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
__constant__ unsigned int cals_Nplane;
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
__constant__ float cBZMxy;
__constant__ float cBZMyx;
__constant__ float cBZPxy;
__constant__ float cBZPyx;
void move_params_device_cals(){
  float tmpp;
  cudaMemcpyToSymbol( cals_SpinSize, &H_SpinSize, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_SpinSize_z, &H_SpinSize_z, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BlockSize_x, &H_BlockSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_BlockSize_y, &H_BlockSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_GridSize_x, &H_GridSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_GridSize_y, &H_GridSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol( cals_Nplane, &H_Nplane, sizeof(unsigned int));
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
  tmpp = (DD);
  cudaMemcpyToSymbol( cBZPxy, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( cBZPyx, &tmpp, sizeof(float));
  tmpp = (-DD);
  cudaMemcpyToSymbol( cBZMxy, &tmpp, sizeof(float));
  tmpp = (DD);
  cudaMemcpyToSymbol( cBZMyx, &tmpp, sizeof(float));
}
__global__ void calTRI(float *confx, float *confy, float *confz, double *out){
	//Energy variables
	extern __shared__ double sD[];
	const int x = threadIdx.x % (cals_BlockSize_x);
	const int y = (threadIdx.x / cals_BlockSize_x);
	const int tx = 3 * (((blockIdx.x % cals_BN) % cals_GridSize_x) * cals_BlockSize_x + x);
	const int ty =(blockIdx.x / cals_BN) * cals_SpinSize * cals_SpinSize_z +  3 * ((((blockIdx.x % cals_BN) / cals_GridSize_x) % cals_GridSize_y) * cals_BlockSize_y + y);
	const int txp = tx +1 ;
	const int typ = ty +1 ;
	const int txp2 = tx +2 ;
	const int typ2 = ty +2 ;
	int z;
	//const int ty = 2 * ((blockIdx.x / cals_BN) * cals_SpinSize + ((blockIdx.x % cals_BN) / cals_GridSize_x) * cals_BlockSize_y + y);
	const int dataoff = (blockIdx.x / cals_BN) * MEASURE_NUM * cals_BN;
	int bx, by, tx_ty = tx + (ty % cals_SpinSize);
	float Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz;
	//-----Calculate the energy of each spin pairs in the system-----
	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.

	bx = (tx + cals_SpinSize - 1) % cals_SpinSize;
	if((ty % cals_SpinSize) == 0)	by = ty + cals_SpinSize - 1;
	else				by = ty - 1;
	//Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.

	z = 0;
	//0,0
	sD[threadIdx.x] = -confx[cals_coo(z, ty, tx)] * ( BXMxx * confx[cals_coo(z, ty, bx)] + BYMxx * confx[cals_coo(z, by, tx)] + BWMxx * confx[cals_coo(z, by, bx)])\
	           -confx[cals_coo(z, ty, tx)] * ( BXMxy * confy[cals_coo(z, ty, bx)] + BYMxy * confy[cals_coo(z, by, tx)] + BWMxy * confy[cals_coo(z, by, bx)])\
	           -confx[cals_coo(z, ty, tx)] * ( cBXMxz * confz[cals_coo(z, ty, bx)] + cBYMxz * confz[cals_coo(z, by, tx)] + cBWMxz * confz[cals_coo(z, by, bx)])\
		         -confy[cals_coo(z, ty, tx)] * ( BXMyx * confx[cals_coo(z, ty, bx)] + BYMyx * confx[cals_coo(z, by, tx)] + BWMyx * confx[cals_coo(z, by, bx)])\
		         -confy[cals_coo(z, ty, tx)] * ( BXMyy * confy[cals_coo(z, ty, bx)] + BYMyy * confy[cals_coo(z, by, tx)] + BWMyy * confy[cals_coo(z, by, bx)])\
		         -confy[cals_coo(z, ty, tx)] * ( cBXMyz * confz[cals_coo(z, ty, bx)] + cBYMyz * confz[cals_coo(z, by, tx)] + cBWMyz * confz[cals_coo(z, by, bx)])\
		         -confz[cals_coo(z, ty, tx)] * ( cBXMzx * confx[cals_coo(z, ty, bx)] + cBYMzx * confx[cals_coo(z, by, tx)] + cBWMzx * confx[cals_coo(z, by, bx)])\
		         -confz[cals_coo(z, ty, tx)] * ( cBXMzy * confy[cals_coo(z, ty, bx)] + cBYMzy * confy[cals_coo(z, by, tx)] + cBWMzy * confy[cals_coo(z, by, bx)])\
		         -confz[cals_coo(z, ty, tx)] * ( BXMzz * confz[cals_coo(z, ty, bx)] + BYMzz * confz[cals_coo(z, by, tx)] + BWMzz * confz[cals_coo(z, by, bx)] - cals_A * confz[cals_coo(z, ty, tx)]);
	//1,0
	sD[threadIdx.x] -= confx[cals_coo(z, typ, tx)] * ( BXMxx * confx[cals_coo(z, typ, bx)] + BYMxx * confx[cals_coo(z, ty, tx)] + BWMxx * confx[cals_coo(z, ty, bx)])\
		         +confx[cals_coo(z, typ, tx)] * ( BXMxy * confy[cals_coo(z, typ, bx)] + BYMxy * confy[cals_coo(z, ty, tx)] + BWMxy * confy[cals_coo(z, ty, bx)])\
		         +confx[cals_coo(z, typ, tx)] * ( cBXMxz * confz[cals_coo(z, typ, bx)] + cBYMxz * confz[cals_coo(z, ty, tx)] + cBWMxz * confz[cals_coo(z, ty, bx)])\
		         +confy[cals_coo(z, typ, tx)] * ( BXMyx * confx[cals_coo(z, typ, bx)] + BYMyx * confx[cals_coo(z, ty, tx)] + BWMyx * confx[cals_coo(z, ty, bx)])\
		         +confy[cals_coo(z, typ, tx)] * ( BXMyy * confy[cals_coo(z, typ, bx)] + BYMyy * confy[cals_coo(z, ty, tx)] + BWMyy * confy[cals_coo(z, ty, bx)])\
		         +confy[cals_coo(z, typ, tx)] * ( cBXMyz * confz[cals_coo(z, typ, bx)] + cBYMyz * confz[cals_coo(z, ty, tx)] + cBWMyz * confz[cals_coo(z, ty, bx)])\
		         +confz[cals_coo(z, typ, tx)] * ( cBXMzx * confx[cals_coo(z, typ, bx)] + cBYMzx * confx[cals_coo(z, ty, tx)] + cBWMzx * confx[cals_coo(z, ty, bx)])\
		         +confz[cals_coo(z, typ, tx)] * ( cBXMzy * confy[cals_coo(z, typ, bx)] + cBYMzy * confy[cals_coo(z, ty, tx)] + cBWMzy * confy[cals_coo(z, ty, bx)])\
		         +confz[cals_coo(z, typ, tx)] * ( BXMzz * confz[cals_coo(z, typ, bx)] + BYMzz * confz[cals_coo(z, ty, tx)] + BWMzz * confz[cals_coo(z, ty, bx)] - cals_A * confz[cals_coo(z, (typ), tx)]);
	//2,0
	sD[threadIdx.x] -= confx[cals_coo(z, typ2, tx)] * ( BXMxx * confx[cals_coo(z, typ2, bx)] + BYMxx * confx[cals_coo(z, typ, tx)] + BWMxx * confx[cals_coo(z, typ, bx)])\
		         +confx[cals_coo(z, typ2, tx)] * ( BXMxy * confy[cals_coo(z, typ2, bx)] + BYMxy * confy[cals_coo(z, typ, tx)] + BWMxy * confy[cals_coo(z, typ, bx)])\
		         +confx[cals_coo(z, typ2, tx)] * ( cBXMxz * confz[cals_coo(z, typ2, bx)] + cBYMxz * confz[cals_coo(z, typ, tx)] + cBWMxz * confz[cals_coo(z, typ, bx)])\
		         +confy[cals_coo(z, typ2, tx)] * ( BXMyx * confx[cals_coo(z, typ2, bx)] + BYMyx * confx[cals_coo(z, typ, tx)] + BWMyx * confx[cals_coo(z, typ, bx)])\
		         +confy[cals_coo(z, typ2, tx)] * ( BXMyy * confy[cals_coo(z, typ2, bx)] + BYMyy * confy[cals_coo(z, typ, tx)] + BWMyy * confy[cals_coo(z, typ, bx)])\
		         +confy[cals_coo(z, typ2, tx)] * ( cBXMyz * confz[cals_coo(z, typ2, bx)] + cBYMyz * confz[cals_coo(z, typ, tx)] + cBWMyz * confz[cals_coo(z, typ, bx)])\
		         +confz[cals_coo(z, typ2, tx)] * ( cBXMzx * confx[cals_coo(z, typ2, bx)] + cBYMzx * confx[cals_coo(z, typ, tx)] + cBWMzx * confx[cals_coo(z, typ, bx)])\
		         +confz[cals_coo(z, typ2, tx)] * ( cBXMzy * confy[cals_coo(z, typ2, bx)] + cBYMzy * confy[cals_coo(z, typ, tx)] + cBWMzy * confy[cals_coo(z, typ, bx)])\
		         +confz[cals_coo(z, typ2, tx)] * ( BXMzz * confz[cals_coo(z, typ2, bx)] + BYMzz * confz[cals_coo(z, typ, tx)] + BWMzz * confz[cals_coo(z, typ, bx)] - cals_A * confz[cals_coo(z, (typ2), tx)]);
	//0,1
	sD[threadIdx.x] -= confx[cals_coo(z, ty, txp)] * ( BXMxx * confx[cals_coo(z, ty, tx)] + BYMxx * confx[cals_coo(z, by, txp)] + BWMxx * confx[cals_coo(z, by, tx)])\
		         +confx[cals_coo(z, ty, txp)] * ( BXMxy * confy[cals_coo(z, ty, tx)] + BYMxy * confy[cals_coo(z, by, txp)] + BWMxy * confy[cals_coo(z, by, tx)])\
		         +confx[cals_coo(z, ty, txp)] * ( cBXMxz * confz[cals_coo(z, ty, tx)] + cBYMxz * confz[cals_coo(z, by, txp)] + cBWMxz * confz[cals_coo(z, by, tx)])\
		         +confy[cals_coo(z, ty, txp)] * ( BXMyx * confx[cals_coo(z, ty, tx)] + BYMyx * confx[cals_coo(z, by, txp)] + BWMyx * confx[cals_coo(z, by, tx)])\
		         +confy[cals_coo(z, ty, txp)] * ( BXMyy * confy[cals_coo(z, ty, tx)] + BYMyy * confy[cals_coo(z, by, txp)] + BWMyy * confy[cals_coo(z, by, tx)])\
		         +confy[cals_coo(z, ty, txp)] * ( cBXMyz * confz[cals_coo(z, ty, tx)] + cBYMyz * confz[cals_coo(z, by, txp)] + cBWMyz * confz[cals_coo(z, by, tx)])\
		         +confz[cals_coo(z, ty, txp)] * ( cBXMzx * confx[cals_coo(z, ty, tx)] + cBYMzx * confx[cals_coo(z, by, txp)] + cBWMzx * confx[cals_coo(z, by, tx)])\
		         +confz[cals_coo(z, ty, txp)] * ( cBXMzy * confy[cals_coo(z, ty, tx)] + cBYMzy * confy[cals_coo(z, by, txp)] + cBWMzy * confy[cals_coo(z, by, tx)])\
		         +confz[cals_coo(z, ty, txp)] * ( BXMzz * confz[cals_coo(z, ty, tx)] + BYMzz * confz[cals_coo(z, by, txp)] + BWMzz * confz[cals_coo(z, by, tx)] - cals_A * confz[cals_coo(z, ty, txp)]);
	//1,1
	sD[threadIdx.x] -= confx[cals_coo(z, typ, txp)] * ( BXMxx * confx[cals_coo(z, typ, tx)] + BYMxx * confx[cals_coo(z, ty, txp)] + BWMxx * confx[cals_coo(z, ty, tx)])\
		         +confx[cals_coo(z, typ, txp)] * ( BXMxy * confy[cals_coo(z, typ, tx)] + BYMxy * confy[cals_coo(z, ty, txp)] + BWMxy * confy[cals_coo(z, ty, tx)])\
		         +confx[cals_coo(z, typ, txp)] * ( cBXMxz * confz[cals_coo(z, typ, tx)] + cBYMxz * confz[cals_coo(z, ty, txp)] + cBWMxz * confz[cals_coo(z, ty, tx)])\
		         +confy[cals_coo(z, typ, txp)] * ( BXMyx * confx[cals_coo(z, typ, tx)] + BYMyx * confx[cals_coo(z, ty, txp)] + BWMyx * confx[cals_coo(z, ty, tx)])\
		         +confy[cals_coo(z, typ, txp)] * ( BXMyy * confy[cals_coo(z, typ, tx)] + BYMyy * confy[cals_coo(z, ty, txp)] + BWMyy * confy[cals_coo(z, ty, tx)])\
		         +confy[cals_coo(z, typ, txp)] * ( cBXMyz * confz[cals_coo(z, typ, tx)] + cBYMyz * confz[cals_coo(z, ty, txp)] + cBWMyz * confz[cals_coo(z, ty, tx)])\
		         +confz[cals_coo(z, typ, txp)] * ( cBXMzx * confx[cals_coo(z, typ, tx)] + cBYMzx * confx[cals_coo(z, ty, txp)] + cBWMzx * confx[cals_coo(z, ty, tx)])\
		         +confz[cals_coo(z, typ, txp)] * ( cBXMzy * confy[cals_coo(z, typ, tx)] + cBYMzy * confy[cals_coo(z, ty, txp)] + cBWMzy * confy[cals_coo(z, ty, tx)])\
		         +confz[cals_coo(z, typ, txp)] * ( BXMzz * confz[cals_coo(z, typ, tx)] + BYMzz * confz[cals_coo(z, ty, txp)] + BWMzz * confz[cals_coo(z, ty, tx)] - cals_A * confz[cals_coo(z, typ, txp)]);
	//2,1
	sD[threadIdx.x] -= confx[cals_coo(z, typ2, txp)] * ( BXMxx * confx[cals_coo(z, typ2, tx)] + BYMxx * confx[cals_coo(z, typ, txp)] + BWMxx * confx[cals_coo(z, typ, tx)])\
		         +confx[cals_coo(z, typ2, txp)] * ( BXMxy * confy[cals_coo(z, typ2, tx)] + BYMxy * confy[cals_coo(z, typ, txp)] + BWMxy * confy[cals_coo(z, typ, tx)])\
		         +confx[cals_coo(z, typ2, txp)] * ( cBXMxz * confz[cals_coo(z, typ2, tx)] + cBYMxz * confz[cals_coo(z, typ, txp)] + cBWMxz * confz[cals_coo(z, typ, tx)])\
		         +confy[cals_coo(z, typ2, txp)] * ( BXMyx * confx[cals_coo(z, typ2, tx)] + BYMyx * confx[cals_coo(z, typ, txp)] + BWMyx * confx[cals_coo(z, typ, tx)])\
		         +confy[cals_coo(z, typ2, txp)] * ( BXMyy * confy[cals_coo(z, typ2, tx)] + BYMyy * confy[cals_coo(z, typ, txp)] + BWMyy * confy[cals_coo(z, typ, tx)])\
		         +confy[cals_coo(z, typ2, txp)] * ( cBXMyz * confz[cals_coo(z, typ2, tx)] + cBYMyz * confz[cals_coo(z, typ, txp)] + cBWMyz * confz[cals_coo(z, typ, tx)])\
		         +confz[cals_coo(z, typ2, txp)] * ( cBXMzx * confx[cals_coo(z, typ2, tx)] + cBYMzx * confx[cals_coo(z, typ, txp)] + cBWMzx * confx[cals_coo(z, typ, tx)])\
		         +confz[cals_coo(z, typ2, txp)] * ( cBXMzy * confy[cals_coo(z, typ2, tx)] + cBYMzy * confy[cals_coo(z, typ, txp)] + cBWMzy * confy[cals_coo(z, typ, tx)])\
		         +confz[cals_coo(z, typ2, txp)] * ( BXMzz * confz[cals_coo(z, typ2, tx)] + BYMzz * confz[cals_coo(z, typ, txp)] + BWMzz * confz[cals_coo(z, typ, tx)] - cals_A * confz[cals_coo(z, typ2, txp)]);
	//0,2
	sD[threadIdx.x] -= confx[cals_coo(z, ty, txp2)] * ( BXMxx * confx[cals_coo(z, ty, txp)] + BYMxx * confx[cals_coo(z, by, txp2)] + BWMxx * confx[cals_coo(z, by, txp)])\
		         +confx[cals_coo(z, ty, txp2)] * ( BXMxy * confy[cals_coo(z, ty, txp)] + BYMxy * confy[cals_coo(z, by, txp2)] + BWMxy * confy[cals_coo(z, by, txp)])\
		         +confx[cals_coo(z, ty, txp2)] * ( cBXMxz * confz[cals_coo(z, ty, txp)] + cBYMxz * confz[cals_coo(z, by, txp2)] + cBWMxz * confz[cals_coo(z, by, txp)])\
		         +confy[cals_coo(z, ty, txp2)] * ( BXMyx * confx[cals_coo(z, ty, txp)] + BYMyx * confx[cals_coo(z, by, txp2)] + BWMyx * confx[cals_coo(z, by, txp)])\
		         +confy[cals_coo(z, ty, txp2)] * ( BXMyy * confy[cals_coo(z, ty, txp)] + BYMyy * confy[cals_coo(z, by, txp2)] + BWMyy * confy[cals_coo(z, by, txp)])\
		         +confy[cals_coo(z, ty, txp2)] * ( cBXMyz * confz[cals_coo(z, ty, txp)] + cBYMyz * confz[cals_coo(z, by, txp2)] + cBWMyz * confz[cals_coo(z, by, txp)])\
		         +confz[cals_coo(z, ty, txp2)] * ( cBXMzx * confx[cals_coo(z, ty, txp)] + cBYMzx * confx[cals_coo(z, by, txp2)] + cBWMzx * confx[cals_coo(z, by, txp)])\
		         +confz[cals_coo(z, ty, txp2)] * ( cBXMzy * confy[cals_coo(z, ty, txp)] + cBYMzy * confy[cals_coo(z, by, txp2)] + cBWMzy * confy[cals_coo(z, by, txp)])\
		         +confz[cals_coo(z, ty, txp2)] * ( BXMzz * confz[cals_coo(z, ty, txp)] + BYMzz * confz[cals_coo(z, by, txp2)] + BWMzz * confz[cals_coo(z, by, txp)] - cals_A * confz[cals_coo(z, ty, txp2)]);
	//1,2
	sD[threadIdx.x] -= confx[cals_coo(z, typ, txp2)] * ( BXMxx * confx[cals_coo(z, typ, txp)] + BYMxx * confx[cals_coo(z, ty, txp2)] + BWMxx * confx[cals_coo(z, ty, txp)])\
		         +confx[cals_coo(z, typ, txp2)] * ( BXMxy * confy[cals_coo(z, typ, txp)] + BYMxy * confy[cals_coo(z, ty, txp2)] + BWMxy * confy[cals_coo(z, ty, txp)])\
		         +confx[cals_coo(z, typ, txp2)] * ( cBXMxz * confz[cals_coo(z, typ, txp)] + cBYMxz * confz[cals_coo(z, ty, txp2)] + cBWMxz * confz[cals_coo(z, ty, txp)])\
		         +confy[cals_coo(z, typ, txp2)] * ( BXMyx * confx[cals_coo(z, typ, txp)] + BYMyx * confx[cals_coo(z, ty, txp2)] + BWMyx * confx[cals_coo(z, ty, txp)])\
		         +confy[cals_coo(z, typ, txp2)] * ( BXMyy * confy[cals_coo(z, typ, txp)] + BYMyy * confy[cals_coo(z, ty, txp2)] + BWMyy * confy[cals_coo(z, ty, txp)])\
		         +confy[cals_coo(z, typ, txp2)] * ( cBXMyz * confz[cals_coo(z, typ, txp)] + cBYMyz * confz[cals_coo(z, ty, txp2)] + cBWMyz * confz[cals_coo(z, ty, txp)])\
		         +confz[cals_coo(z, typ, txp2)] * ( cBXMzx * confx[cals_coo(z, typ, txp)] + cBYMzx * confx[cals_coo(z, ty, txp2)] + cBWMzx * confx[cals_coo(z, ty, txp)])\
		         +confz[cals_coo(z, typ, txp2)] * ( cBXMzy * confy[cals_coo(z, typ, txp)] + cBYMzy * confy[cals_coo(z, ty, txp2)] + cBWMzy * confy[cals_coo(z, ty, txp)])\
		         +confz[cals_coo(z, typ, txp2)] * ( BXMzz * confz[cals_coo(z, typ, txp)] + BYMzz * confz[cals_coo(z, ty, txp2)] + BWMzz * confz[cals_coo(z, ty, txp)] - cals_A * confz[cals_coo(z, typ, txp2)]);
	//2,2
	sD[threadIdx.x] -= confx[cals_coo(z, typ2, txp2)] * ( BXMxx * confx[cals_coo(z, typ2, txp)] + BYMxx * confx[cals_coo(z, typ, txp2)] + BWMxx * confx[cals_coo(z, typ, txp)])\
		         +confx[cals_coo(z, typ2, txp2)] * ( BXMxy * confy[cals_coo(z, typ2, txp)] + BYMxy * confy[cals_coo(z, typ, txp2)] + BWMxy * confy[cals_coo(z, typ, txp)])\
		         +confx[cals_coo(z, typ2, txp2)] * ( cBXMxz * confz[cals_coo(z, typ2, txp)] + cBYMxz * confz[cals_coo(z, typ, txp2)] + cBWMxz * confz[cals_coo(z, typ, txp)])\
		         +confy[cals_coo(z, typ2, txp2)] * ( BXMyx * confx[cals_coo(z, typ2, txp)] + BYMyx * confx[cals_coo(z, typ, txp2)] + BWMyx * confx[cals_coo(z, typ, txp)])\
		         +confy[cals_coo(z, typ2, txp2)] * ( BXMyy * confy[cals_coo(z, typ2, txp)] + BYMyy * confy[cals_coo(z, typ, txp2)] + BWMyy * confy[cals_coo(z, typ, txp)])\
		         +confy[cals_coo(z, typ2, txp2)] * ( cBXMyz * confz[cals_coo(z, typ2, txp)] + cBYMyz * confz[cals_coo(z, typ, txp2)] + cBWMyz * confz[cals_coo(z, typ, txp)])\
		         +confz[cals_coo(z, typ2, txp2)] * ( cBXMzx * confx[cals_coo(z, typ2, txp)] + cBYMzx * confx[cals_coo(z, typ, txp2)] + cBWMzx * confx[cals_coo(z, typ, txp)])\
		         +confz[cals_coo(z, typ2, txp2)] * ( cBXMzy * confy[cals_coo(z, typ2, txp)] + cBYMzy * confy[cals_coo(z, typ, txp2)] + cBWMzy * confy[cals_coo(z, typ, txp)])\
		         +confz[cals_coo(z, typ2, txp2)] * ( BXMzz * confz[cals_coo(z, typ2, txp)] + BYMzz * confz[cals_coo(z, typ, txp2)] + BWMzz * confz[cals_coo(z, typ, txp)] - cals_A * confz[cals_coo(z, typ2, txp2)]);
  for (z = 1; z < cals_SpinSize_z; z++){
	//0,0
	sD[threadIdx.x] = -confx[cals_coo(z, ty, tx)] * ( BXMxx * confx[cals_coo(z, ty, bx)] + BYMxx * confx[cals_coo(z, by, tx)] + BWMxx * confx[cals_coo(z, by, bx)] + BZMxx * confx[cals_coo(z-1, ty, tx)])\
	           -confx[cals_coo(z, ty, tx)] * ( BXMxy * confy[cals_coo(z, ty, bx)] + BYMxy * confy[cals_coo(z, by, tx)] + BWMxy * confy[cals_coo(z, by, bx)] + cBZMxy * confy[cals_coo(z-1, ty, tx)])\
	           -confx[cals_coo(z, ty, tx)] * ( cBXMxz * confz[cals_coo(z, ty, bx)] + cBYMxz * confz[cals_coo(z, by, tx)] + cBWMxz * confz[cals_coo(z, by, bx)])\
		         -confy[cals_coo(z, ty, tx)] * ( BXMyx * confx[cals_coo(z, ty, bx)] + BYMyx * confx[cals_coo(z, by, tx)] + BWMyx * confx[cals_coo(z, by, bx)] + cBZMyx * confx[cals_coo(z-1, ty, tx)])\
		         -confy[cals_coo(z, ty, tx)] * ( BXMyy * confy[cals_coo(z, ty, bx)] + BYMyy * confy[cals_coo(z, by, tx)] + BWMyy * confy[cals_coo(z, by, bx)] + BZMyy * confy[cals_coo(z-1, ty, tx)])\
		         -confy[cals_coo(z, ty, tx)] * ( cBXMyz * confz[cals_coo(z, ty, bx)] + cBYMyz * confz[cals_coo(z, by, tx)] + cBWMyz * confz[cals_coo(z, by, bx)])\
		         -confz[cals_coo(z, ty, tx)] * ( cBXMzx * confx[cals_coo(z, ty, bx)] + cBYMzx * confx[cals_coo(z, by, tx)] + cBWMzx * confx[cals_coo(z, by, bx)])\
		         -confz[cals_coo(z, ty, tx)] * ( cBXMzy * confy[cals_coo(z, ty, bx)] + cBYMzy * confy[cals_coo(z, by, tx)] + cBWMzy * confy[cals_coo(z, by, bx)])\
		         -confz[cals_coo(z, ty, tx)] * ( BXMzz * confz[cals_coo(z, ty, bx)] + BYMzz * confz[cals_coo(z, by, tx)] + BWMzz * confz[cals_coo(z, by, bx)] + BZMzz * confz[cals_coo(z-1, ty, tx)] - cals_A * confz[cals_coo(z, ty, tx)]);
	//1,0
	sD[threadIdx.x] -= confx[cals_coo(z, typ, tx)] * ( BXMxx * confx[cals_coo(z, typ, bx)] + BYMxx * confx[cals_coo(z, ty, tx)] + BWMxx * confx[cals_coo(z, ty, bx)] + BZMxx * confx[cals_coo(z-1, typ, tx)])\
		         +confx[cals_coo(z, typ, tx)] * ( BXMxy * confy[cals_coo(z, typ, bx)] + BYMxy * confy[cals_coo(z, ty, tx)] + BWMxy * confy[cals_coo(z, ty, bx)] + cBZMxy * confy[cals_coo(z-1, typ, tx)])\
		         +confx[cals_coo(z, typ, tx)] * ( cBXMxz * confz[cals_coo(z, typ, bx)] + cBYMxz * confz[cals_coo(z, ty, tx)] + cBWMxz * confz[cals_coo(z, ty, bx)])\
		         +confy[cals_coo(z, typ, tx)] * ( BXMyx * confx[cals_coo(z, typ, bx)] + BYMyx * confx[cals_coo(z, ty, tx)] + BWMyx * confx[cals_coo(z, ty, bx)] + cBZMyx * confx[cals_coo(z-1, typ, tx)])\
		         +confy[cals_coo(z, typ, tx)] * ( BXMyy * confy[cals_coo(z, typ, bx)] + BYMyy * confy[cals_coo(z, ty, tx)] + BWMyy * confy[cals_coo(z, ty, bx)] + BZMyy * confy[cals_coo(z-1, typ, tx)])\
		         +confy[cals_coo(z, typ, tx)] * ( cBXMyz * confz[cals_coo(z, typ, bx)] + cBYMyz * confz[cals_coo(z, ty, tx)] + cBWMyz * confz[cals_coo(z, ty, bx)])\
		         +confz[cals_coo(z, typ, tx)] * ( cBXMzx * confx[cals_coo(z, typ, bx)] + cBYMzx * confx[cals_coo(z, ty, tx)] + cBWMzx * confx[cals_coo(z, ty, bx)])\
		         +confz[cals_coo(z, typ, tx)] * ( cBXMzy * confy[cals_coo(z, typ, bx)] + cBYMzy * confy[cals_coo(z, ty, tx)] + cBWMzy * confy[cals_coo(z, ty, bx)])\
		         +confz[cals_coo(z, typ, tx)] * ( BXMzz * confz[cals_coo(z, typ, bx)] + BYMzz * confz[cals_coo(z, ty, tx)] + BWMzz * confz[cals_coo(z, ty, bx)] + BZMzz * confz[cals_coo(z-1, typ, tx)] - cals_A * confz[cals_coo(z, (typ), tx)]);
	//2,0
	sD[threadIdx.x] -= confx[cals_coo(z, typ2, tx)] * ( BXMxx * confx[cals_coo(z, typ2, bx)] + BYMxx * confx[cals_coo(z, typ, tx)] + BWMxx * confx[cals_coo(z, typ, bx)] + BZMxx * confx[cals_coo(z-1, typ2, tx)])\
		         +confx[cals_coo(z, typ2, tx)] * ( BXMxy * confy[cals_coo(z, typ2, bx)] + BYMxy * confy[cals_coo(z, typ, tx)] + BWMxy * confy[cals_coo(z, typ, bx)] + cBZMxy * confy[cals_coo(z-1, typ2, tx)])\
		         +confx[cals_coo(z, typ2, tx)] * ( cBXMxz * confz[cals_coo(z, typ2, bx)] + cBYMxz * confz[cals_coo(z, typ, tx)] + cBWMxz * confz[cals_coo(z, typ, bx)])\
		         +confy[cals_coo(z, typ2, tx)] * ( BXMyx * confx[cals_coo(z, typ2, bx)] + BYMyx * confx[cals_coo(z, typ, tx)] + BWMyx * confx[cals_coo(z, typ, bx)] + cBZMyx * confx[cals_coo(z-1, typ2, tx)])\
		         +confy[cals_coo(z, typ2, tx)] * ( BXMyy * confy[cals_coo(z, typ2, bx)] + BYMyy * confy[cals_coo(z, typ, tx)] + BWMyy * confy[cals_coo(z, typ, bx)] + BZMyy * confy[cals_coo(z-1, typ2, tx)])\
		         +confy[cals_coo(z, typ2, tx)] * ( cBXMyz * confz[cals_coo(z, typ2, bx)] + cBYMyz * confz[cals_coo(z, typ, tx)] + cBWMyz * confz[cals_coo(z, typ, bx)])\
		         +confz[cals_coo(z, typ2, tx)] * ( cBXMzx * confx[cals_coo(z, typ2, bx)] + cBYMzx * confx[cals_coo(z, typ, tx)] + cBWMzx * confx[cals_coo(z, typ, bx)])\
		         +confz[cals_coo(z, typ2, tx)] * ( cBXMzy * confy[cals_coo(z, typ2, bx)] + cBYMzy * confy[cals_coo(z, typ, tx)] + cBWMzy * confy[cals_coo(z, typ, bx)])\
		         +confz[cals_coo(z, typ2, tx)] * ( BXMzz * confz[cals_coo(z, typ2, bx)] + BYMzz * confz[cals_coo(z, typ, tx)] + BWMzz * confz[cals_coo(z, typ, bx)] + BZMzz * confz[cals_coo(z-1, typ2, tx)] - cals_A * confz[cals_coo(z, (typ2), tx)]);
	//0,1
	sD[threadIdx.x] -= confx[cals_coo(z, ty, txp)] * ( BXMxx * confx[cals_coo(z, ty, tx)] + BYMxx * confx[cals_coo(z, by, txp)] + BWMxx * confx[cals_coo(z, by, tx)] + BZMxx * confx[cals_coo(z-1, ty, txp)])\
		         +confx[cals_coo(z, ty, txp)] * ( BXMxy * confy[cals_coo(z, ty, tx)] + BYMxy * confy[cals_coo(z, by, txp)] + BWMxy * confy[cals_coo(z, by, tx)] + cBZMxy * confy[cals_coo(z-1, ty, txp)])\
		         +confx[cals_coo(z, ty, txp)] * ( cBXMxz * confz[cals_coo(z, ty, tx)] + cBYMxz * confz[cals_coo(z, by, txp)] + cBWMxz * confz[cals_coo(z, by, tx)])\
		         +confy[cals_coo(z, ty, txp)] * ( BXMyx * confx[cals_coo(z, ty, tx)] + BYMyx * confx[cals_coo(z, by, txp)] + BWMyx * confx[cals_coo(z, by, tx)] + cBZMyx * confx[cals_coo(z-1, ty, txp)])\
		         +confy[cals_coo(z, ty, txp)] * ( BXMyy * confy[cals_coo(z, ty, tx)] + BYMyy * confy[cals_coo(z, by, txp)] + BWMyy * confy[cals_coo(z, by, tx)] + BZMyy * confy[cals_coo(z-1, ty, txp)])\
		         +confy[cals_coo(z, ty, txp)] * ( cBXMyz * confz[cals_coo(z, ty, tx)] + cBYMyz * confz[cals_coo(z, by, txp)] + cBWMyz * confz[cals_coo(z, by, tx)])\
		         +confz[cals_coo(z, ty, txp)] * ( cBXMzx * confx[cals_coo(z, ty, tx)] + cBYMzx * confx[cals_coo(z, by, txp)] + cBWMzx * confx[cals_coo(z, by, tx)])\
		         +confz[cals_coo(z, ty, txp)] * ( cBXMzy * confy[cals_coo(z, ty, tx)] + cBYMzy * confy[cals_coo(z, by, txp)] + cBWMzy * confy[cals_coo(z, by, tx)])\
		         +confz[cals_coo(z, ty, txp)] * ( BXMzz * confz[cals_coo(z, ty, tx)] + BYMzz * confz[cals_coo(z, by, txp)] + BWMzz * confz[cals_coo(z, by, tx)] + BZMzz * confz[cals_coo(z-1, ty, txp)] - cals_A * confz[cals_coo(z, ty, txp)]);
	//1,1
	sD[threadIdx.x] -= confx[cals_coo(z, typ, txp)] * ( BXMxx * confx[cals_coo(z, typ, tx)] + BYMxx * confx[cals_coo(z, ty, txp)] + BWMxx * confx[cals_coo(z, ty, tx)] + BZMxx * confx[cals_coo(z-1, typ, txp)])\
		         +confx[cals_coo(z, typ, txp)] * ( BXMxy * confy[cals_coo(z, typ, tx)] + BYMxy * confy[cals_coo(z, ty, txp)] + BWMxy * confy[cals_coo(z, ty, tx)] + cBZMxy * confy[cals_coo(z-1, typ, txp)])\
		         +confx[cals_coo(z, typ, txp)] * ( cBXMxz * confz[cals_coo(z, typ, tx)] + cBYMxz * confz[cals_coo(z, ty, txp)] + cBWMxz * confz[cals_coo(z, ty, tx)])\
		         +confy[cals_coo(z, typ, txp)] * ( BXMyx * confx[cals_coo(z, typ, tx)] + BYMyx * confx[cals_coo(z, ty, txp)] + BWMyx * confx[cals_coo(z, ty, tx)] + cBZMyx * confx[cals_coo(z-1, typ, txp)])\
		         +confy[cals_coo(z, typ, txp)] * ( BXMyy * confy[cals_coo(z, typ, tx)] + BYMyy * confy[cals_coo(z, ty, txp)] + BWMyy * confy[cals_coo(z, ty, tx)] + BZMyy * confy[cals_coo(z-1, typ, txp)])\
		         +confy[cals_coo(z, typ, txp)] * ( cBXMyz * confz[cals_coo(z, typ, tx)] + cBYMyz * confz[cals_coo(z, ty, txp)] + cBWMyz * confz[cals_coo(z, ty, tx)])\
		         +confz[cals_coo(z, typ, txp)] * ( cBXMzx * confx[cals_coo(z, typ, tx)] + cBYMzx * confx[cals_coo(z, ty, txp)] + cBWMzx * confx[cals_coo(z, ty, tx)])\
		         +confz[cals_coo(z, typ, txp)] * ( cBXMzy * confy[cals_coo(z, typ, tx)] + cBYMzy * confy[cals_coo(z, ty, txp)] + cBWMzy * confy[cals_coo(z, ty, tx)])\
		         +confz[cals_coo(z, typ, txp)] * ( BXMzz * confz[cals_coo(z, typ, tx)] + BYMzz * confz[cals_coo(z, ty, txp)] + BWMzz * confz[cals_coo(z, ty, tx)] + BZMzz * confz[cals_coo(z-1, typ, txp)] - cals_A * confz[cals_coo(z, typ, txp)]);
	//2,1
	sD[threadIdx.x] -= confx[cals_coo(z, typ2, txp)] * ( BXMxx * confx[cals_coo(z, typ2, tx)] + BYMxx * confx[cals_coo(z, typ, txp)] + BWMxx * confx[cals_coo(z, typ, tx)] + BZMxx * confx[cals_coo(z-1, typ2, txp)])\
		         +confx[cals_coo(z, typ2, txp)] * ( BXMxy * confy[cals_coo(z, typ2, tx)] + BYMxy * confy[cals_coo(z, typ, txp)] + BWMxy * confy[cals_coo(z, typ, tx)] + cBZMxy * confy[cals_coo(z-1, typ2, txp)])\
		         +confx[cals_coo(z, typ2, txp)] * ( cBXMxz * confz[cals_coo(z, typ2, tx)] + cBYMxz * confz[cals_coo(z, typ, txp)] + cBWMxz * confz[cals_coo(z, typ, tx)])\
		         +confy[cals_coo(z, typ2, txp)] * ( BXMyx * confx[cals_coo(z, typ2, tx)] + BYMyx * confx[cals_coo(z, typ, txp)] + BWMyx * confx[cals_coo(z, typ, tx)] + cBZMyx * confx[cals_coo(z-1, typ2, txp)])\
		         +confy[cals_coo(z, typ2, txp)] * ( BXMyy * confy[cals_coo(z, typ2, tx)] + BYMyy * confy[cals_coo(z, typ, txp)] + BWMyy * confy[cals_coo(z, typ, tx)] + BZMyy * confy[cals_coo(z-1, typ2, txp)])\
		         +confy[cals_coo(z, typ2, txp)] * ( cBXMyz * confz[cals_coo(z, typ2, tx)] + cBYMyz * confz[cals_coo(z, typ, txp)] + cBWMyz * confz[cals_coo(z, typ, tx)])\
		         +confz[cals_coo(z, typ2, txp)] * ( cBXMzx * confx[cals_coo(z, typ2, tx)] + cBYMzx * confx[cals_coo(z, typ, txp)] + cBWMzx * confx[cals_coo(z, typ, tx)])\
		         +confz[cals_coo(z, typ2, txp)] * ( cBXMzy * confy[cals_coo(z, typ2, tx)] + cBYMzy * confy[cals_coo(z, typ, txp)] + cBWMzy * confy[cals_coo(z, typ, tx)])\
		         +confz[cals_coo(z, typ2, txp)] * ( BXMzz * confz[cals_coo(z, typ2, tx)] + BYMzz * confz[cals_coo(z, typ, txp)] + BWMzz * confz[cals_coo(z, typ, tx)] + BZMzz * confz[cals_coo(z-1, typ2, txp)] - cals_A * confz[cals_coo(z, typ2, txp)]);
	//0,2
	sD[threadIdx.x] -= confx[cals_coo(z, ty, txp2)] * ( BXMxx * confx[cals_coo(z, ty, txp)] + BYMxx * confx[cals_coo(z, by, txp2)] + BWMxx * confx[cals_coo(z, by, txp)] + BZMxx * confx[cals_coo(z-1, ty, txp2)])\
		         +confx[cals_coo(z, ty, txp2)] * ( BXMxy * confy[cals_coo(z, ty, txp)] + BYMxy * confy[cals_coo(z, by, txp2)] + BWMxy * confy[cals_coo(z, by, txp)] + cBZMxy * confy[cals_coo(z-1, ty, txp2)])\
		         +confx[cals_coo(z, ty, txp2)] * ( cBXMxz * confz[cals_coo(z, ty, txp)] + cBYMxz * confz[cals_coo(z, by, txp2)] + cBWMxz * confz[cals_coo(z, by, txp)])\
		         +confy[cals_coo(z, ty, txp2)] * ( BXMyx * confx[cals_coo(z, ty, txp)] + BYMyx * confx[cals_coo(z, by, txp2)] + BWMyx * confx[cals_coo(z, by, txp)] + cBZMyx * confx[cals_coo(z-1, ty, txp2)])\
		         +confy[cals_coo(z, ty, txp2)] * ( BXMyy * confy[cals_coo(z, ty, txp)] + BYMyy * confy[cals_coo(z, by, txp2)] + BWMyy * confy[cals_coo(z, by, txp)] + BZMyy * confy[cals_coo(z-1, ty, txp2)])\
		         +confy[cals_coo(z, ty, txp2)] * ( cBXMyz * confz[cals_coo(z, ty, txp)] + cBYMyz * confz[cals_coo(z, by, txp2)] + cBWMyz * confz[cals_coo(z, by, txp)])\
		         +confz[cals_coo(z, ty, txp2)] * ( cBXMzx * confx[cals_coo(z, ty, txp)] + cBYMzx * confx[cals_coo(z, by, txp2)] + cBWMzx * confx[cals_coo(z, by, txp)])\
		         +confz[cals_coo(z, ty, txp2)] * ( cBXMzy * confy[cals_coo(z, ty, txp)] + cBYMzy * confy[cals_coo(z, by, txp2)] + cBWMzy * confy[cals_coo(z, by, txp)])\
		         +confz[cals_coo(z, ty, txp2)] * ( BXMzz * confz[cals_coo(z, ty, txp)] + BYMzz * confz[cals_coo(z, by, txp2)] + BWMzz * confz[cals_coo(z, by, txp)] + BZMzz * confz[cals_coo(z-1, ty, txp2)] - cals_A * confz[cals_coo(z, ty, txp2)]);
	//1,2
	sD[threadIdx.x] -= confx[cals_coo(z, typ, txp2)] * ( BXMxx * confx[cals_coo(z, typ, txp)] + BYMxx * confx[cals_coo(z, ty, txp2)] + BWMxx * confx[cals_coo(z, ty, txp)] + BZMxx * confx[cals_coo(z-1, typ, txp2)])\
		         +confx[cals_coo(z, typ, txp2)] * ( BXMxy * confy[cals_coo(z, typ, txp)] + BYMxy * confy[cals_coo(z, ty, txp2)] + BWMxy * confy[cals_coo(z, ty, txp)] + cBZMxy * confy[cals_coo(z-1, typ, txp2)])\
		         +confx[cals_coo(z, typ, txp2)] * ( cBXMxz * confz[cals_coo(z, typ, txp)] + cBYMxz * confz[cals_coo(z, ty, txp2)] + cBWMxz * confz[cals_coo(z, ty, txp)])\
		         +confy[cals_coo(z, typ, txp2)] * ( BXMyx * confx[cals_coo(z, typ, txp)] + BYMyx * confx[cals_coo(z, ty, txp2)] + BWMyx * confx[cals_coo(z, ty, txp)] + cBZMyx * confx[cals_coo(z-1, typ, txp2)])\
		         +confy[cals_coo(z, typ, txp2)] * ( BXMyy * confy[cals_coo(z, typ, txp)] + BYMyy * confy[cals_coo(z, ty, txp2)] + BWMyy * confy[cals_coo(z, ty, txp)] + BZMyy * confy[cals_coo(z-1, typ, txp2)])\
		         +confy[cals_coo(z, typ, txp2)] * ( cBXMyz * confz[cals_coo(z, typ, txp)] + cBYMyz * confz[cals_coo(z, ty, txp2)] + cBWMyz * confz[cals_coo(z, ty, txp)])\
		         +confz[cals_coo(z, typ, txp2)] * ( cBXMzx * confx[cals_coo(z, typ, txp)] + cBYMzx * confx[cals_coo(z, ty, txp2)] + cBWMzx * confx[cals_coo(z, ty, txp)])\
		         +confz[cals_coo(z, typ, txp2)] * ( cBXMzy * confy[cals_coo(z, typ, txp)] + cBYMzy * confy[cals_coo(z, ty, txp2)] + cBWMzy * confy[cals_coo(z, ty, txp)])\
		         +confz[cals_coo(z, typ, txp2)] * ( BXMzz * confz[cals_coo(z, typ, txp)] + BYMzz * confz[cals_coo(z, ty, txp2)] + BWMzz * confz[cals_coo(z, ty, txp)] + BZMzz * confz[cals_coo(z-1, typ, txp2)] - cals_A * confz[cals_coo(z, typ, txp2)]);
	//2,2
	sD[threadIdx.x] -= confx[cals_coo(z, typ2, txp2)] * ( BXMxx * confx[cals_coo(z, typ2, txp)] + BYMxx * confx[cals_coo(z, typ, txp2)] + BWMxx * confx[cals_coo(z, typ, txp)] + BZMxx * confx[cals_coo(z-1, typ2, txp2)])\
		         +confx[cals_coo(z, typ2, txp2)] * ( BXMxy * confy[cals_coo(z, typ2, txp)] + BYMxy * confy[cals_coo(z, typ, txp2)] + BWMxy * confy[cals_coo(z, typ, txp)] + cBZMxy * confy[cals_coo(z-1, typ2, txp2)])\
		         +confx[cals_coo(z, typ2, txp2)] * ( cBXMxz * confz[cals_coo(z, typ2, txp)] + cBYMxz * confz[cals_coo(z, typ, txp2)] + cBWMxz * confz[cals_coo(z, typ, txp)])\
		         +confy[cals_coo(z, typ2, txp2)] * ( BXMyx * confx[cals_coo(z, typ2, txp)] + BYMyx * confx[cals_coo(z, typ, txp2)] + BWMyx * confx[cals_coo(z, typ, txp)] + cBZMyx * confx[cals_coo(z-1, typ2, txp2)])\
		         +confy[cals_coo(z, typ2, txp2)] * ( BXMyy * confy[cals_coo(z, typ2, txp)] + BYMyy * confy[cals_coo(z, typ, txp2)] + BWMyy * confy[cals_coo(z, typ, txp)] + BZMyy * confy[cals_coo(z-1, typ2, txp2)])\
		         +confy[cals_coo(z, typ2, txp2)] * ( cBXMyz * confz[cals_coo(z, typ2, txp)] + cBYMyz * confz[cals_coo(z, typ, txp2)] + cBWMyz * confz[cals_coo(z, typ, txp)])\
		         +confz[cals_coo(z, typ2, txp2)] * ( cBXMzx * confx[cals_coo(z, typ2, txp)] + cBYMzx * confx[cals_coo(z, typ, txp2)] + cBWMzx * confx[cals_coo(z, typ, txp)])\
		         +confz[cals_coo(z, typ2, txp2)] * ( cBXMzy * confy[cals_coo(z, typ2, txp)] + cBYMzy * confy[cals_coo(z, typ, txp2)] + cBWMzy * confy[cals_coo(z, typ, txp)])\
		         +confz[cals_coo(z, typ2, txp2)] * ( BXMzz * confz[cals_coo(z, typ2, txp)] + BYMzz * confz[cals_coo(z, typ, txp2)] + BWMzz * confz[cals_coo(z, typ, txp)] + BZMzz * confz[cals_coo(z-1, typ2, txp2)] - cals_A * confz[cals_coo(z, typ2, txp2)]);
	}
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
  for (z = 0; z < cals_SpinSize_z; z++){
		sD[threadIdx.x]  = confx[cals_coo(z, ty, tx)];
		sD[threadIdx.x] += confx[cals_coo(z, typ, tx)];
		sD[threadIdx.x] += confx[cals_coo(z, typ2, tx)];
		sD[threadIdx.x] += confx[cals_coo(z, ty, txp)];
		sD[threadIdx.x] += confx[cals_coo(z, typ, txp)];
		sD[threadIdx.x] += confx[cals_coo(z, typ2, txp)];
		sD[threadIdx.x] += confx[cals_coo(z, ty, txp2)];
		sD[threadIdx.x] += confx[cals_coo(z, typ, txp2)];
		sD[threadIdx.x] += confx[cals_coo(z, typ2, txp2)];
	}
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
  for (z = 0; z < cals_SpinSize_z; z++){
		sD[threadIdx.x]  = confy[cals_coo(z, ty, tx)];
		sD[threadIdx.x] += confy[cals_coo(z, typ, tx)];
		sD[threadIdx.x] += confy[cals_coo(z, typ2, tx)];
		sD[threadIdx.x] += confy[cals_coo(z, ty, txp)];
		sD[threadIdx.x] += confy[cals_coo(z, typ, txp)];
		sD[threadIdx.x] += confy[cals_coo(z, typ2, txp)];
		sD[threadIdx.x] += confy[cals_coo(z, ty, txp2)];
		sD[threadIdx.x] += confy[cals_coo(z, typ, txp2)];
		sD[threadIdx.x] += confy[cals_coo(z, typ2, txp2)];
	}
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
  for (z = 0; z < cals_SpinSize_z; z++){
		sD[threadIdx.x]  = confz[cals_coo(z, ty, tx)];
		sD[threadIdx.x] += confz[cals_coo(z, typ, tx)];
		sD[threadIdx.x] += confz[cals_coo(z, typ2, tx)];
		sD[threadIdx.x] += confz[cals_coo(z, ty, txp)];
		sD[threadIdx.x] += confz[cals_coo(z, typ, txp)];
		sD[threadIdx.x] += confz[cals_coo(z, typ2, txp)];
		sD[threadIdx.x] += confz[cals_coo(z, ty, txp2)];
		sD[threadIdx.x] += confz[cals_coo(z, typ, txp2)];
		sD[threadIdx.x] += confz[cals_coo(z, typ2, txp2)];
	}
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
	z=0;
	Ax = confx[cals_coo(z, ty, tx)];
	Ay = confy[cals_coo(z, ty, tx)];
	Az = confz[cals_coo(z, ty, tx)];
	Bx = confx[cals_coo(z, ty, bx)];
	By = confy[cals_coo(z, ty, bx)];
	Bz = confz[cals_coo(z, ty, bx)];
	Cx = confx[cals_coo(z, by, bx)];
	Cy = confy[cals_coo(z, by, bx)];
	Cz = confz[cals_coo(z, by, bx)];
	sD[threadIdx.x] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, by, tx)];
	By = confy[cals_coo(z, by, tx)];
	Bz = confz[cals_coo(z, by, tx)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(1,0)
	Ax = confx[cals_coo(z, typ, tx)];
	Ay = confy[cals_coo(z, typ, tx)];
	Az = confz[cals_coo(z, typ, tx)];
	Bx = confx[cals_coo(z, typ, bx)];
	By = confy[cals_coo(z, typ, bx)];
	Bz = confz[cals_coo(z, typ, bx)];
	Cx = confx[cals_coo(z, ty, bx)];
	Cy = confy[cals_coo(z, ty, bx)];
	Cz = confz[cals_coo(z, ty, bx)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, ty, tx)];
	By = confy[cals_coo(z, ty, tx)];
	Bz = confz[cals_coo(z, ty, tx)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(2,0)
	Ax = confx[cals_coo(z, typ2, tx)];
	Ay = confy[cals_coo(z, typ2, tx)];
	Az = confz[cals_coo(z, typ2, tx)];
	Bx = confx[cals_coo(z, typ2, bx)];
	By = confy[cals_coo(z, typ2, bx)];
	Bz = confz[cals_coo(z, typ2, bx)];
	Cx = confx[cals_coo(z, typ, bx)];
	Cy = confy[cals_coo(z, typ, bx)];
	Cz = confz[cals_coo(z, typ, bx)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, typ, tx)];
	By = confy[cals_coo(z, typ, tx)];
	Bz = confz[cals_coo(z, typ, tx)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(0,1)
	Ax = confx[cals_coo(z, ty, txp)];
	Ay = confy[cals_coo(z, ty, txp)];
	Az = confz[cals_coo(z, ty, txp)];
	Bx = confx[cals_coo(z, ty, tx)];
	By = confy[cals_coo(z, ty, tx)];
	Bz = confz[cals_coo(z, ty, tx)];
	Cx = confx[cals_coo(z, by, tx)];
	Cy = confy[cals_coo(z, by, tx)];
	Cz = confz[cals_coo(z, by, tx)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, by, txp)];
	By = confy[cals_coo(z, by, txp)];
	Bz = confz[cals_coo(z, by, txp)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(1,1)
	Ax = confx[cals_coo(z, typ, txp)];
	Ay = confy[cals_coo(z, typ, txp)];
	Az = confz[cals_coo(z, typ, txp)];
	Bx = confx[cals_coo(z, typ, tx)];
	By = confy[cals_coo(z, typ, tx)];
	Bz = confz[cals_coo(z, typ, tx)];
	Cx = confx[cals_coo(z, ty, tx)];
	Cy = confy[cals_coo(z, ty, tx)];
	Cz = confz[cals_coo(z, ty, tx)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, ty, txp)];
	By = confy[cals_coo(z, ty, txp)];
	Bz = confz[cals_coo(z, ty, txp)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(2,1)
	Ax = confx[cals_coo(z, typ2, txp)];
	Ay = confy[cals_coo(z, typ2, txp)];
	Az = confz[cals_coo(z, typ2, txp)];
	Bx = confx[cals_coo(z, typ2, tx)];
	By = confy[cals_coo(z, typ2, tx)];
	Bz = confz[cals_coo(z, typ2, tx)];
	Cx = confx[cals_coo(z, typ, tx)];
	Cy = confy[cals_coo(z, typ, tx)];
	Cz = confz[cals_coo(z, typ, tx)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, typ, txp)];
	By = confy[cals_coo(z, typ, txp)];
	Bz = confz[cals_coo(z, typ, txp)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(0,2)
	Ax = confx[cals_coo(z, ty, txp2)];
	Ay = confy[cals_coo(z, ty, txp2)];
	Az = confz[cals_coo(z, ty, txp2)];
	Bx = confx[cals_coo(z, ty, txp)];
	By = confy[cals_coo(z, ty, txp)];
	Bz = confz[cals_coo(z, ty, txp)];
	Cx = confx[cals_coo(z, by, txp)];
	Cy = confy[cals_coo(z, by, txp)];
	Cz = confz[cals_coo(z, by, txp)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, by, txp2)];
	By = confy[cals_coo(z, by, txp2)];
	Bz = confz[cals_coo(z, by, txp2)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(1,2)
	Ax = confx[cals_coo(z, typ, txp2)];
	Ay = confy[cals_coo(z, typ, txp2)];
	Az = confz[cals_coo(z, typ, txp2)];
	Bx = confx[cals_coo(z, typ, txp)];
	By = confy[cals_coo(z, typ, txp)];
	Bz = confz[cals_coo(z, typ, txp)];
	Cx = confx[cals_coo(z, ty, txp)];
	Cy = confy[cals_coo(z, ty, txp)];
	Cz = confz[cals_coo(z, ty, txp)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, ty, txp2)];
	By = confy[cals_coo(z, ty, txp2)];
	Bz = confz[cals_coo(z, ty, txp2)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(2,2)
	Ax = confx[cals_coo(z, typ2, txp2)];
	Ay = confy[cals_coo(z, typ2, txp2)];
	Az = confz[cals_coo(z, typ2, txp2)];
	Bx = confx[cals_coo(z, typ2, txp)];
	By = confy[cals_coo(z, typ2, txp)];
	Bz = confz[cals_coo(z, typ2, txp)];
	Cx = confx[cals_coo(z, typ, txp)];
	Cy = confy[cals_coo(z, typ, txp)];
	Cz = confz[cals_coo(z, typ, txp)];
	sD[threadIdx.x] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[cals_coo(z, typ, txp2)];
	By = confy[cals_coo(z, typ, txp2)];
	Bz = confz[cals_coo(z, typ, txp2)];
	sD[threadIdx.x] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
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
	sD[threadIdx.x]  = confx[cals_coo(z, ty, tx)]     * cosf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, tx)]    * cosf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, tx)]   * cosf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp)]    * cosf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp)]   * cosf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp)]  * cosf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp2)]   * cosf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp2)]  * cosf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp2)] * cosf(Q1x*(txp2) + Q1y*(typ2));
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
	sD[threadIdx.x]  = confy[cals_coo(z, ty, tx)]     * cosf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, tx)]    * cosf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, tx)]   * cosf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp)]    * cosf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp)]   * cosf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp)]  * cosf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp2)]   * cosf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp2)]  * cosf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp2)] * cosf(Q1x*(txp2) + Q1y*(typ2));
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
	sD[threadIdx.x]  = confz[cals_coo(z, ty, tx)]     * cosf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, tx)]    * cosf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, tx)]   * cosf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp)]    * cosf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp)]   * cosf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp)]  * cosf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp2)]   * cosf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp2)]  * cosf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp2)] * cosf(Q1x*(txp2) + Q1y*(typ2));
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
	sD[threadIdx.x]  = confx[cals_coo(z, ty, tx)]     * sinf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, tx)]    * sinf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, tx)]   * sinf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp)]    * sinf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp)]   * sinf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp)]  * sinf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp2)]   * sinf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp2)]  * sinf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp2)] * sinf(Q1x*(txp2) + Q1y*(typ2));
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
	sD[threadIdx.x]  = confy[cals_coo(z, ty, tx)]     * sinf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, tx)]    * sinf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, tx)]   * sinf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp)]    * sinf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp)]   * sinf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp)]  * sinf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp2)]   * sinf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp2)]  * sinf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp2)] * sinf(Q1x*(txp2) + Q1y*(typ2));
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
	sD[threadIdx.x]  = confz[cals_coo(z, ty, tx)]     * sinf(Q1x*(tx  ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, tx)]    * sinf(Q1x*(tx  ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, tx)]   * sinf(Q1x*(tx  ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp)]    * sinf(Q1x*(txp ) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp)]   * sinf(Q1x*(txp ) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp)]  * sinf(Q1x*(txp ) + Q1y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp2)]   * sinf(Q1x*(txp2) + Q1y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp2)]  * sinf(Q1x*(txp2) + Q1y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp2)] * sinf(Q1x*(txp2) + Q1y*(typ2));
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
	sD[threadIdx.x]  = confx[cals_coo(z, ty, tx)]     * cosf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, tx)]    * cosf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, tx)]   * cosf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp)]    * cosf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp)]   * cosf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp)]  * cosf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp2)]   * cosf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp2)]  * cosf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp2)] * cosf(Q2x*(txp2) + Q2y*(typ2));
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
	sD[threadIdx.x]  = confy[cals_coo(z, ty, tx)]     * cosf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, tx)]    * cosf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, tx)]   * cosf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp)]    * cosf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp)]   * cosf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp)]  * cosf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp2)]   * cosf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp2)]  * cosf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp2)] * cosf(Q2x*(txp2) + Q2y*(typ2));
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
	sD[threadIdx.x]  = confz[cals_coo(z, ty, tx)]     * cosf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, tx)]    * cosf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, tx)]   * cosf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp)]    * cosf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp)]   * cosf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp)]  * cosf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp2)]   * cosf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp2)]  * cosf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp2)] * cosf(Q2x*(txp2) + Q2y*(typ2));
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
	sD[threadIdx.x]  = confx[cals_coo(z, ty, tx)]     * sinf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, tx)]    * sinf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, tx)]   * sinf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp)]    * sinf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp)]   * sinf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp)]  * sinf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confx[cals_coo(z, ty, txp2)]   * sinf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confx[cals_coo(z, typ, txp2)]  * sinf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confx[cals_coo(z, typ2, txp2)] * sinf(Q2x*(txp2) + Q2y*(typ2));
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
	sD[threadIdx.x]  = confy[cals_coo(z, ty, tx)]     * sinf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, tx)]    * sinf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, tx)]   * sinf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp)]    * sinf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp)]   * sinf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp)]  * sinf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confy[cals_coo(z, ty, txp2)]   * sinf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confy[cals_coo(z, typ, txp2)]  * sinf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confy[cals_coo(z, typ2, txp2)] * sinf(Q2x*(txp2) + Q2y*(typ2));
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
	sD[threadIdx.x]  = confz[cals_coo(z, ty, tx)]     * sinf(Q2x*(tx  ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, tx)]    * sinf(Q2x*(tx  ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, tx)]   * sinf(Q2x*(tx  ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp)]    * sinf(Q2x*(txp ) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp)]   * sinf(Q2x*(txp ) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp)]  * sinf(Q2x*(txp ) + Q2y*(typ2));
	sD[threadIdx.x] += confz[cals_coo(z, ty, txp2)]   * sinf(Q2x*(txp2) + Q2y*(ty  ));
	sD[threadIdx.x] += confz[cals_coo(z, typ, txp2)]  * sinf(Q2x*(txp2) + Q2y*(typ ));
	sD[threadIdx.x] += confz[cals_coo(z, typ2, txp2)] * sinf(Q2x*(txp2) + Q2y*(typ2));
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
