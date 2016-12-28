#include "measurements.cuh"
#ifdef TRI
__constant__ unsigned int corr_SpinSize;
__constant__ unsigned int corr_SpinSize_z;
__constant__ unsigned int corr_BlockSize_x;
__constant__ unsigned int corr_BlockSize_y;
__constant__ unsigned int corr_GridSize_x;
__constant__ unsigned int corr_GridSize_y;
__constant__ unsigned int corr_N;
__constant__ unsigned int corr_Nplane;
__constant__ unsigned int corr_TN;
__constant__ unsigned int corr_BN;
void move_params_device_corr(){
  cudaMemcpyToSymbol(corr_SpinSize, &H_SpinSize, sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_SpinSize_z, &H_SpinSize_z, sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_BlockSize_x, &H_BlockSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_BlockSize_y, &H_BlockSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_GridSize_x, &H_GridSize_x, sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_GridSize_y, &H_GridSize_y, sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_N , &H_N , sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_Nplane , &H_Nplane , sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_TN, &H_TN, sizeof(unsigned int));
  cudaMemcpyToSymbol(corr_BN, &H_BN, sizeof(unsigned int));
}
__global__ void skyr_den_gen(const float *confx, const float *confy, const float *confz, float *skyr_den){
	//Energy variables
	const int x = threadIdx.x % (corr_BlockSize_x);
	const int y = (threadIdx.x / corr_BlockSize_x);
	const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
	const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
	const int txp = tx +1 ;
	const int typ = ty +1 ;
	const int txp2 = tx +2 ;
	const int typ2 = ty +2 ;
	int typ3 = ty +3 ;
	int txp3 = (tx + 3) % corr_SpinSize;
	//const int ty = 2 * ((blockIdx.x / cals_BN) * cals_SpinSize + ((blockIdx.x % cals_BN) / cals_GridSize_x) * cals_BlockSize_y + y);
	const int dataoff = (blockIdx.x / corr_BN) * MEASURE_NUM * corr_BN;
	int bx, by, tx_ty = tx + (ty % corr_SpinSize);
	float Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz;
	//-----Calculate the energy of each spin pairs in the system-----
	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.

	bx = (tx + corr_SpinSize - 1) % corr_SpinSize;
	if((ty % corr_SpinSize) == 0)	by = ty + corr_SpinSize - 1;
	else				by = ty - 1;
	if((typ3 % corr_SpinSize) == 0)	typ3 = typ3 - corr_SpinSize;
	//try to measure Chern number
	Ax = confx[corr_coo2D(ty, tx)];
	Ay = confy[corr_coo2D(ty, tx)];
	Az = confz[corr_coo2D(ty, tx)];
	Bx = confx[corr_coo2D(by, bx)];
	By = confy[corr_coo2D(by, bx)];
	Bz = confz[corr_coo2D(by, bx)];
	Cx = confx[corr_coo2D(ty, bx)];
	Cy = confy[corr_coo2D(ty, bx)];
	Cz = confz[corr_coo2D(ty, bx)];
	skyr_den[corr_coo2D(ty, tx)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(by, tx)];
	Cy = confy[corr_coo2D(by, tx)];
	Cz = confz[corr_coo2D(by, tx)];
	skyr_den[corr_coo2D(ty, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(ty, txp)];
	By = confy[corr_coo2D(ty, txp)];
	Bz = confz[corr_coo2D(ty, txp)];
	skyr_den[corr_coo2D(ty, tx)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(ty, txp)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(by, txp)];
	Ay = confy[corr_coo2D(by, txp)];
	Az = confz[corr_coo2D(by, txp)];
	skyr_den[corr_coo2D(ty, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(ty, txp2)];
	Cy = confy[corr_coo2D(ty, txp2)];
	Cz = confz[corr_coo2D(ty, txp2)];
	skyr_den[corr_coo2D(ty, txp)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(ty, txp2)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(by, txp2)];
	By = confy[corr_coo2D(by, txp2)];
	Bz = confz[corr_coo2D(by, txp2)];
	skyr_den[corr_coo2D(ty, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(ty, txp3)];
	Ay = confy[corr_coo2D(ty, txp3)];
	Az = confz[corr_coo2D(ty, txp3)];
	skyr_den[corr_coo2D(ty, txp2)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(1,0)
	Ax = confx[corr_coo2D(typ, tx)];
	Ay = confy[corr_coo2D(typ, tx)];
	Az = confz[corr_coo2D(typ, tx)];
	Bx = confx[corr_coo2D(ty, bx)];
	By = confy[corr_coo2D(ty, bx)];
	Bz = confz[corr_coo2D(ty, bx)];
	Cx = confx[corr_coo2D(typ, bx)];
	Cy = confy[corr_coo2D(typ, bx)];
	Cz = confz[corr_coo2D(typ, bx)];
	skyr_den[corr_coo2D(typ, tx)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(ty, tx)];
	Cy = confy[corr_coo2D(ty, tx)];
	Cz = confz[corr_coo2D(ty, tx)];
	skyr_den[corr_coo2D(ty, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(typ, txp)];
	By = confy[corr_coo2D(typ, txp)];
	Bz = confz[corr_coo2D(typ, txp)];
	skyr_den[corr_coo2D(ty, tx)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, tx)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, txp)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(ty, txp)];
	Ay = confy[corr_coo2D(ty, txp)];
	Az = confz[corr_coo2D(ty, txp)];
	skyr_den[corr_coo2D(ty, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(ty, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(typ, txp2)];
	Cy = confy[corr_coo2D(typ, txp2)];
	Cz = confz[corr_coo2D(typ, txp2)];
	skyr_den[corr_coo2D(ty, txp)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, txp)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, txp2)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(ty, txp2)];
	By = confy[corr_coo2D(ty, txp2)];
	Bz = confz[corr_coo2D(ty, txp2)];
	skyr_den[corr_coo2D(ty, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(ty, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(typ, txp3)];
	Ay = confy[corr_coo2D(typ, txp3)];
	Az = confz[corr_coo2D(typ, txp3)];
	skyr_den[corr_coo2D(ty, txp2)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, txp2)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(ty, txp3)];
	Cy = confy[corr_coo2D(ty, txp3)];
	Cz = confz[corr_coo2D(ty, txp3)];
	skyr_den[corr_coo2D(ty, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(2,0)
	Ax = confx[corr_coo2D(typ2, tx)];
	Ay = confy[corr_coo2D(typ2, tx)];
	Az = confz[corr_coo2D(typ2, tx)];
	Bx = confx[corr_coo2D(typ, bx)];
	By = confy[corr_coo2D(typ, bx)];
	Bz = confz[corr_coo2D(typ, bx)];
	Cx = confx[corr_coo2D(typ2, bx)];
	Cy = confy[corr_coo2D(typ2, bx)];
	Cz = confz[corr_coo2D(typ2, bx)];
	skyr_den[corr_coo2D(typ2, tx)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(typ, tx)];
	Cy = confy[corr_coo2D(typ, tx)];
	Cz = confz[corr_coo2D(typ, tx)];
	skyr_den[corr_coo2D(typ, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(typ2, txp)];
	By = confy[corr_coo2D(typ2, txp)];
	Bz = confz[corr_coo2D(typ2, txp)];
	skyr_den[corr_coo2D(typ, tx)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, tx)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, txp)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(typ, txp)];
	Ay = confy[corr_coo2D(typ, txp)];
	Az = confz[corr_coo2D(typ, txp)];
	skyr_den[corr_coo2D(typ, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(typ2, txp2)];
	Cy = confy[corr_coo2D(typ2, txp2)];
	Cz = confz[corr_coo2D(typ2, txp2)];
	skyr_den[corr_coo2D(typ, txp)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, txp)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, txp2)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(typ, txp2)];
	By = confy[corr_coo2D(typ, txp2)];
	Bz = confz[corr_coo2D(typ, txp2)];
	skyr_den[corr_coo2D(typ, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(typ2, txp3)];
	Ay = confy[corr_coo2D(typ2, txp3)];
	Az = confz[corr_coo2D(typ2, txp3)];
	skyr_den[corr_coo2D(typ, txp2)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, txp2)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(typ, txp3)];
	Cy = confy[corr_coo2D(typ, txp3)];
	Cz = confz[corr_coo2D(typ, txp3)];
	skyr_den[corr_coo2D(typ, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	//(3,0)
	Ax = confx[corr_coo2D(typ3, tx)];
	Ay = confy[corr_coo2D(typ3, tx)];
	Az = confz[corr_coo2D(typ3, tx)];
	Bx = confx[corr_coo2D(typ2, bx)];
	By = confy[corr_coo2D(typ2, bx)];
	Bz = confz[corr_coo2D(typ2, bx)];
	Cx = confx[corr_coo2D(typ2, tx)];
	Cy = confy[corr_coo2D(typ2, tx)];
	Cz = confz[corr_coo2D(typ2, tx)];
	skyr_den[corr_coo2D(typ2, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(typ3, txp)];
	By = confy[corr_coo2D(typ3, txp)];
	Bz = confz[corr_coo2D(typ3, txp)];
	skyr_den[corr_coo2D(typ2, tx)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(typ2, txp)];
	Ay = confy[corr_coo2D(typ2, txp)];
	Az = confz[corr_coo2D(typ2, txp)];
	skyr_den[corr_coo2D(typ2, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(typ3, txp2)];
	Cy = confy[corr_coo2D(typ3, txp2)];
	Cz = confz[corr_coo2D(typ3, txp2)];
	skyr_den[corr_coo2D(typ2, txp)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Bx = confx[corr_coo2D(typ2, txp2)];
	By = confy[corr_coo2D(typ2, txp2)];
	Bz = confz[corr_coo2D(typ2, txp2)];
	skyr_den[corr_coo2D(typ2, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	skyr_den[corr_coo2D(typ2, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Ax = confx[corr_coo2D(typ3, txp3)];
	Ay = confy[corr_coo2D(typ3, txp3)];
	Az = confz[corr_coo2D(typ3, txp3)];
	skyr_den[corr_coo2D(typ2, txp2)] += 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	Cx = confx[corr_coo2D(typ2, txp3)];
	Cy = confy[corr_coo2D(typ2, txp3)];
	Cz = confz[corr_coo2D(typ2, txp3)];
	skyr_den[corr_coo2D(typ2, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
	__syncthreads();
	skyr_den[corr_coo2D(ty, tx)] /= 3.0;
	skyr_den[corr_coo2D(ty, txp)] /= 3.0;
	skyr_den[corr_coo2D(ty, txp2)] /= 3.0;
	skyr_den[corr_coo2D(typ, tx)] /= 3.0;
	skyr_den[corr_coo2D(typ, txp)] /= 3.0;
	skyr_den[corr_coo2D(typ, txp2)] /= 3.0;
	skyr_den[corr_coo2D(typ2, tx)] /= 3.0;
	skyr_den[corr_coo2D(typ2, txp)] /= 3.0;
	skyr_den[corr_coo2D(typ2, txp2)] /= 3.0;
	__syncthreads();
}
//__global__ void skyr_den_gen(const float *confx, const float *confy, const float *confz, float *skyr_den){
//	//Energy variables
//	const int x = threadIdx.x % (corr_BlockSize_x);
//	const int y = (threadIdx.x / corr_BlockSize_x);
//	const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
//	const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
//	const int txp = tx +1 ;
//	const int typ = ty +1 ;
//	const int txp2 = tx +2 ;
//	const int typ2 = ty +2 ;
//	//const int ty = 2 * ((blockIdx.x / cals_BN) * cals_SpinSize + ((blockIdx.x % cals_BN) / cals_GridSize_x) * cals_BlockSize_y + y);
//	const int dataoff = (blockIdx.x / corr_BN) * MEASURE_NUM * corr_BN;
//	int bx, by, tx_ty = tx + (ty % corr_SpinSize);
//	float Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz;
//	//-----Calculate the energy of each spin pairs in the system-----
//	//To avoid double counting, for each spin, choose the neighbor spin on the left hand side of each spin and also one above each spin as pairs. Each spin has two pairs.
//
//	bx = (tx + corr_SpinSize - 1) % corr_SpinSize;
//	if((ty % corr_SpinSize) == 0)	by = ty + corr_SpinSize - 1;
//	else				by = ty - 1;
//	//try to measure Chern number
//	//(0,0)
//	Ax = confx[corr_coo2D(ty, tx)];
//	Ay = confy[corr_coo2D(ty, tx)];
//	Az = confz[corr_coo2D(ty, tx)];
//	Bx = confx[corr_coo2D(ty, bx)];
//	By = confy[corr_coo2D(ty, bx)];
//	Bz = confz[corr_coo2D(ty, bx)];
//	Cx = confx[corr_coo2D(by, bx)];
//	Cy = confy[corr_coo2D(by, bx)];
//	Cz = confz[corr_coo2D(by, bx)];
//	skyr_den[corr_coo2D(ty, tx)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(by, tx)];
//	By = confy[corr_coo2D(by, tx)];
//	Bz = confz[corr_coo2D(by, tx)];
//	skyr_den[corr_coo2D(ty, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(1,0)
//	Ax = confx[corr_coo2D(typ, tx)];
//	Ay = confy[corr_coo2D(typ, tx)];
//	Az = confz[corr_coo2D(typ, tx)];
//	Bx = confx[corr_coo2D(typ, bx)];
//	By = confy[corr_coo2D(typ, bx)];
//	Bz = confz[corr_coo2D(typ, bx)];
//	Cx = confx[corr_coo2D(ty, bx)];
//	Cy = confy[corr_coo2D(ty, bx)];
//	Cz = confz[corr_coo2D(ty, bx)];
//	skyr_den[corr_coo2D(typ, tx)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(ty, tx)];
//	By = confy[corr_coo2D(ty, tx)];
//	Bz = confz[corr_coo2D(ty, tx)];
//	skyr_den[corr_coo2D(typ, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(2,0)
//	Ax = confx[corr_coo2D(typ2, tx)];
//	Ay = confy[corr_coo2D(typ2, tx)];
//	Az = confz[corr_coo2D(typ2, tx)];
//	Bx = confx[corr_coo2D(typ2, bx)];
//	By = confy[corr_coo2D(typ2, bx)];
//	Bz = confz[corr_coo2D(typ2, bx)];
//	Cx = confx[corr_coo2D(typ, bx)];
//	Cy = confy[corr_coo2D(typ, bx)];
//	Cz = confz[corr_coo2D(typ, bx)];
//	skyr_den[corr_coo2D(typ2, tx)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(typ, tx)];
//	By = confy[corr_coo2D(typ, tx)];
//	Bz = confz[corr_coo2D(typ, tx)];
//	skyr_den[corr_coo2D(typ2, tx)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(0,1)
//	Ax = confx[corr_coo2D(ty, txp)];
//	Ay = confy[corr_coo2D(ty, txp)];
//	Az = confz[corr_coo2D(ty, txp)];
//	Bx = confx[corr_coo2D(ty, tx)];
//	By = confy[corr_coo2D(ty, tx)];
//	Bz = confz[corr_coo2D(ty, tx)];
//	Cx = confx[corr_coo2D(by, tx)];
//	Cy = confy[corr_coo2D(by, tx)];
//	Cz = confz[corr_coo2D(by, tx)];
//	skyr_den[corr_coo2D(ty, txp)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(by, txp)];
//	By = confy[corr_coo2D(by, txp)];
//	Bz = confz[corr_coo2D(by, txp)];
//	skyr_den[corr_coo2D(ty, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(1,1)
//	Ax = confx[corr_coo2D(typ, txp)];
//	Ay = confy[corr_coo2D(typ, txp)];
//	Az = confz[corr_coo2D(typ, txp)];
//	Bx = confx[corr_coo2D(typ, tx)];
//	By = confy[corr_coo2D(typ, tx)];
//	Bz = confz[corr_coo2D(typ, tx)];
//	Cx = confx[corr_coo2D(ty, tx)];
//	Cy = confy[corr_coo2D(ty, tx)];
//	Cz = confz[corr_coo2D(ty, tx)];
//	skyr_den[corr_coo2D(typ, txp)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(ty, txp)];
//	By = confy[corr_coo2D(ty, txp)];
//	Bz = confz[corr_coo2D(ty, txp)];
//	skyr_den[corr_coo2D(typ, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(2,1)
//	Ax = confx[corr_coo2D(typ2, txp)];
//	Ay = confy[corr_coo2D(typ2, txp)];
//	Az = confz[corr_coo2D(typ2, txp)];
//	Bx = confx[corr_coo2D(typ2, tx)];
//	By = confy[corr_coo2D(typ2, tx)];
//	Bz = confz[corr_coo2D(typ2, tx)];
//	Cx = confx[corr_coo2D(typ, tx)];
//	Cy = confy[corr_coo2D(typ, tx)];
//	Cz = confz[corr_coo2D(typ, tx)];
//	skyr_den[corr_coo2D(typ2, txp)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(typ, txp)];
//	By = confy[corr_coo2D(typ, txp)];
//	Bz = confz[corr_coo2D(typ, txp)];
//	skyr_den[corr_coo2D(typ2, txp)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(0,2)
//	Ax = confx[corr_coo2D(ty, txp2)];
//	Ay = confy[corr_coo2D(ty, txp2)];
//	Az = confz[corr_coo2D(ty, txp2)];
//	Bx = confx[corr_coo2D(ty, txp)];
//	By = confy[corr_coo2D(ty, txp)];
//	Bz = confz[corr_coo2D(ty, txp)];
//	Cx = confx[corr_coo2D(by, txp)];
//	Cy = confy[corr_coo2D(by, txp)];
//	Cz = confz[corr_coo2D(by, txp)];
//	skyr_den[corr_coo2D(ty, txp2)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(by, txp2)];
//	By = confy[corr_coo2D(by, txp2)];
//	Bz = confz[corr_coo2D(by, txp2)];
//	skyr_den[corr_coo2D(ty, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(1,2)
//	Ax = confx[corr_coo2D(typ, txp2)];
//	Ay = confy[corr_coo2D(typ, txp2)];
//	Az = confz[corr_coo2D(typ, txp2)];
//	Bx = confx[corr_coo2D(typ, txp)];
//	By = confy[corr_coo2D(typ, txp)];
//	Bz = confz[corr_coo2D(typ, txp)];
//	Cx = confx[corr_coo2D(ty, txp)];
//	Cy = confy[corr_coo2D(ty, txp)];
//	Cz = confz[corr_coo2D(ty, txp)];
//	skyr_den[corr_coo2D(typ, txp2)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(ty, txp2)];
//	By = confy[corr_coo2D(ty, txp2)];
//	Bz = confz[corr_coo2D(ty, txp2)];
//	skyr_den[corr_coo2D(typ, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	//(2,2)
//	Ax = confx[corr_coo2D(typ2, txp2)];
//	Ay = confy[corr_coo2D(typ2, txp2)];
//	Az = confz[corr_coo2D(typ2, txp2)];
//	Bx = confx[corr_coo2D(typ2, txp)];
//	By = confy[corr_coo2D(typ2, txp)];
//	Bz = confz[corr_coo2D(typ2, txp)];
//	Cx = confx[corr_coo2D(typ, txp)];
//	Cy = confy[corr_coo2D(typ, txp)];
//	Cz = confz[corr_coo2D(typ, txp)];
//	skyr_den[corr_coo2D(typ2, txp2)] = 2*atan((Ax * (By*Cz-Bz*Cy) + Ay * (Bz*Cx-Bx*Cz) + Az * (Bx*Cy-By*Cx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	Bx = confx[corr_coo2D(typ, txp2)];
//	By = confy[corr_coo2D(typ, txp2)];
//	Bz = confz[corr_coo2D(typ, txp2)];
//	skyr_den[corr_coo2D(typ2, txp2)] += 2*atan((Ax * (Cy*Bz-Cz*By) + Ay * (Cz*Bx-Cx*Bz) + Az * (Cx*By-Cy*Bx))/
//	  (1.0 + Ax*Bx + Ay*By + Az*Bz + Cx*Bx + Cy*By + Cz*Bz + Ax*Cx + Ay*Cy + Az*Cz));
//	__syncthreads();
//}
__global__ void getcorrTRI(const float *confx, float *corr, int original_i, int original_j){
  /*****************************************************************
    !!!!!!!!!!!!!!! It can be used for square lattice and triangular lattice.
    Set ( original_i, original_j) as our original point.
    for tx_o , ty_o in 2x2 block of (original_i, original_j):
    corr[i - tx_o][j - ty_o] <-  the correlation between  and  (i, j)
    corr[   tx   ][   ty   ]
    use the periodic condition to keep the index positive.
    We need to sum over different (original_i, original_j) to get the correlation.
   *****************************************************************/
  //Energy variables
  const int x = threadIdx.x % (corr_BlockSize_x);
  const int y = (threadIdx.x / corr_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
  const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
  const int ox = original_i;
  const int oy =(blockIdx.x / corr_BN) * corr_SpinSize + original_j;
  //const int txp = tx +1 ;
  //const int typ = ty +1 ;
  //const int ty = 2 * ((blockIdx.x / BN) * SpinSize + ((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
  float sx00, sx01, sx02,
        sx10, sx11, sx12,
        sx20, sx21, sx22;
  int fx0, fy0,
      fx1, fy1,
      fx2, fy2,
      fx3, fy3,//from o to f
      fx4, fy4;
  //calculate all the final position first

  fx0 = (tx + original_i) % corr_SpinSize;
  fx1 = (tx + original_i + 1) % corr_SpinSize;
  fx2 = (tx + original_i + 2) % corr_SpinSize;
  fx3 = (tx + original_i + 3) % corr_SpinSize;
  fx4 = (tx + original_i + 4) % corr_SpinSize;

  if((ty % corr_SpinSize + original_j) >= corr_SpinSize)	fy0 = ty + original_j - corr_SpinSize;
  else  fy0 = ty + original_j;
  if((ty % corr_SpinSize + original_j + 1) >= corr_SpinSize)	fy1 = ty + original_j + 1 - corr_SpinSize;
  else  fy1 = ty + original_j + 1;
  if((ty % corr_SpinSize + original_j + 2) >= corr_SpinSize)	fy2 = ty + original_j + 2 - corr_SpinSize;
  else  fy2 = ty + original_j + 2;
  if((ty % corr_SpinSize + original_j + 3) >= corr_SpinSize)	fy3 = ty + original_j + 3 - corr_SpinSize;
  else  fy3 = ty + original_j + 3;
  if((ty % corr_SpinSize + original_j + 4) >= corr_SpinSize)	fy4 = ty + original_j + 4 - corr_SpinSize;
  else  fy4 = ty + original_j + 4;

  //Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.
  sx00 = confx[corr_coo2D(oy,ox)];
  sx01 = confx[corr_coo2D(oy,ox+1)];
  sx02 = confx[corr_coo2D(oy,ox+2)];
  sx10 = confx[corr_coo2D(oy+1,ox)];
  sx11 = confx[corr_coo2D(oy+1,ox+1)];
  sx12 = confx[corr_coo2D(oy+1,ox+2)];
  sx20 = confx[corr_coo2D(oy+2,ox)];
  sx21 = confx[corr_coo2D(oy+2,ox+1)];
  sx22 = confx[corr_coo2D(oy+2,ox+2)];
  corr[corr_coo2D(ty,tx)] += sx00 * confx[corr_coo2D( fy0,fx0)] +
                        sx01 * confx[corr_coo2D( fy0,fx1)] +
                        sx02 * confx[corr_coo2D( fy0,fx2)] +
                        sx10 * confx[corr_coo2D( fy1,fx0)] +
                        sx11 * confx[corr_coo2D( fy1,fx1)] +
                        sx12 * confx[corr_coo2D( fy1,fx2)] +
                        sx20 * confx[corr_coo2D( fy2,fx0)] +
                        sx21 * confx[corr_coo2D( fy2,fx1)] +
                        sx22 * confx[corr_coo2D( fy2,fx2)] ;
  corr[corr_coo2D(ty,tx+1)] += sx00 * confx[corr_coo2D( fy0,fx1)] +
                          sx01 * confx[corr_coo2D( fy0,fx2)] +
                          sx02 * confx[corr_coo2D( fy0,fx3)] +
                          sx10 * confx[corr_coo2D( fy1,fx1)] +
                          sx11 * confx[corr_coo2D( fy1,fx2)] +
                          sx12 * confx[corr_coo2D( fy1,fx3)] +
                          sx20 * confx[corr_coo2D( fy2,fx1)] +
                          sx21 * confx[corr_coo2D( fy2,fx2)] +
                          sx22 * confx[corr_coo2D( fy2,fx3)] ;
  corr[corr_coo2D(ty,tx+2)] += sx00 * confx[corr_coo2D( fy0,fx2)] +
                          sx01 * confx[corr_coo2D( fy0,fx3)] +
                          sx02 * confx[corr_coo2D( fy0,fx4)] +
                          sx10 * confx[corr_coo2D( fy1,fx2)] +
                          sx11 * confx[corr_coo2D( fy1,fx3)] +
                          sx12 * confx[corr_coo2D( fy1,fx4)] +
                          sx20 * confx[corr_coo2D( fy2,fx2)] +
                          sx21 * confx[corr_coo2D( fy2,fx3)] +
                          sx22 * confx[corr_coo2D( fy2,fx4)] ;
  corr[corr_coo2D((ty+1),tx)] += sx00 * confx[corr_coo2D( fy1,fx0)] +
                            sx01 * confx[corr_coo2D( fy1,fx1)] +
                            sx02 * confx[corr_coo2D( fy1,fx2)] +
                            sx10 * confx[corr_coo2D( fy2,fx0)] +
                            sx11 * confx[corr_coo2D( fy2,fx1)] +
                            sx12 * confx[corr_coo2D( fy2,fx2)] +
                            sx20 * confx[corr_coo2D( fy3,fx0)] +
                            sx21 * confx[corr_coo2D( fy3,fx1)] +
                            sx22 * confx[corr_coo2D( fy3,fx2)] ;
  corr[corr_coo2D((ty+1),tx+1)] += sx00 * confx[corr_coo2D( fy1,fx1)] +
                              sx01 * confx[corr_coo2D( fy1,fx2)] +
                              sx02 * confx[corr_coo2D( fy1,fx3)] +
                              sx10 * confx[corr_coo2D( fy2,fx1)] +
                              sx11 * confx[corr_coo2D( fy2,fx2)] +
                              sx12 * confx[corr_coo2D( fy2,fx3)] +
                              sx20 * confx[corr_coo2D( fy3,fx1)] +
                              sx21 * confx[corr_coo2D( fy3,fx2)] +
                              sx22 * confx[corr_coo2D( fy3,fx3)] ;
  corr[corr_coo2D((ty+1),tx+2)] += sx00 * confx[corr_coo2D( fy1,fx2)] +
                              sx01 * confx[corr_coo2D( fy1,fx3)] +
                              sx02 * confx[corr_coo2D( fy1,fx4)] +
                              sx10 * confx[corr_coo2D( fy2,fx2)] +
                              sx11 * confx[corr_coo2D( fy2,fx3)] +
                              sx12 * confx[corr_coo2D( fy2,fx4)] +
                              sx20 * confx[corr_coo2D( fy3,fx2)] +
                              sx21 * confx[corr_coo2D( fy3,fx3)] +
                              sx22 * confx[corr_coo2D( fy3,fx4)] ;
  corr[corr_coo2D((ty+2),tx)] += sx00 * confx[corr_coo2D( fy2,fx0)] +
                            sx01 * confx[corr_coo2D( fy2,fx1)] +
                            sx02 * confx[corr_coo2D( fy2,fx2)] +
                            sx10 * confx[corr_coo2D( fy3,fx0)] +
                            sx11 * confx[corr_coo2D( fy3,fx1)] +
                            sx12 * confx[corr_coo2D( fy3,fx2)] +
                            sx20 * confx[corr_coo2D( fy4,fx0)] +
                            sx21 * confx[corr_coo2D( fy4,fx1)] +
                            sx22 * confx[corr_coo2D( fy4,fx2)] ;
  corr[corr_coo2D((ty+2),tx+1)] += sx00 * confx[corr_coo2D( fy2,fx1)] +
                              sx01 * confx[corr_coo2D( fy2,fx2)] +
                              sx02 * confx[corr_coo2D( fy2,fx3)] +
                              sx10 * confx[corr_coo2D( fy3,fx1)] +
                              sx11 * confx[corr_coo2D( fy3,fx2)] +
                              sx12 * confx[corr_coo2D( fy3,fx3)] +
                              sx20 * confx[corr_coo2D( fy4,fx1)] +
                              sx21 * confx[corr_coo2D( fy4,fx2)] +
                              sx22 * confx[corr_coo2D( fy4,fx3)] ;
  corr[corr_coo2D((ty+2),tx+2)] += sx00 * confx[corr_coo2D( fy2,fx2)] +
                              sx01 * confx[corr_coo2D( fy2,fx3)] +
                              sx02 * confx[corr_coo2D( fy2,fx4)] +
                              sx10 * confx[corr_coo2D( fy3,fx2)] +
                              sx11 * confx[corr_coo2D( fy3,fx3)] +
                              sx12 * confx[corr_coo2D( fy3,fx4)] +
                              sx20 * confx[corr_coo2D( fy4,fx2)] +
                              sx21 * confx[corr_coo2D( fy4,fx3)] +
                              sx22 * confx[corr_coo2D( fy4,fx4)] ;
  __syncthreads();
}
__global__ void getFTskyr(const float *confx, float *corr, int original_i, int original_j){
  /*****************************************************************
    !!!!!!!!!!!!!!! It can be used for square lattice and triangular lattice.
    Set ( original_i, original_j) as our original point.
    for tx_o , ty_o in 2x2 block of (original_i, original_j):
    corr[i - tx_o][j - ty_o] <-  the correlation between  and  (i, j)
    corr[   tx   ][   ty   ]
    use the periodic condition to keep the index positive.
    We need to sum over different (original_i, original_j) to get the correlation.
   *****************************************************************/
  //Energy variables
  const int x = threadIdx.x % (corr_BlockSize_x);
  const int y = (threadIdx.x / corr_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
  const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
  const int ox = original_i;
  const int oy =(blockIdx.x / corr_BN) * corr_SpinSize + original_j;
  //const int txp = tx +1 ;
  //const int typ = ty +1 ;
  //const int ty = 2 * ((blockIdx.x / BN) * SpinSize + ((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
  float sx00, sx01, sx02,
        sx10, sx11, sx12,
        sx20, sx21, sx22;
  float fx0, fy0, fx1, fy1, fx2, fy2;
  int ox0 = (ox+0)%corr_SpinSize;
  int ox1 = (ox+1)%corr_SpinSize;
  int ox2 = (ox+2)%corr_SpinSize;
  int ox3 = (ox+3)%corr_SpinSize;
  int oy0 = (oy+0)%corr_SpinSize;
  int oy1 = (oy+1)%corr_SpinSize;
  int oy2 = (oy+2)%corr_SpinSize;
  int oy3 = (oy+3)%corr_SpinSize;
  float sinshift = (tx>=(corr_SpinSize/2))?(3.0/4.0*TWOPI):0.0;
  //calculate all the final position first

  fx0 = (tx) % (corr_SpinSize/2);
  fx1 = (tx + 1) % (corr_SpinSize/2);
  fx2 = (tx + 2) % (corr_SpinSize/2);
  fx0 *= TWOPI/float(corr_SpinSize);
  fx1 *= TWOPI/float(corr_SpinSize);
  fx2 *= TWOPI/float(corr_SpinSize);

  fy0 = (ty) % corr_SpinSize;
  fy1 = (ty + 1) % corr_SpinSize;
  fy2 = (ty + 2) % corr_SpinSize;
  fy0 *= TWOPI/float(corr_SpinSize);
  fy1 *= TWOPI/float(corr_SpinSize);
  fy2 *= TWOPI/float(corr_SpinSize);


  //Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.
  sx00 = confx[corr_coo2D(oy  , ox  )];
  sx01 = confx[corr_coo2D(oy  , ox+1)];
  sx02 = confx[corr_coo2D(oy  , ox+2)];
  sx10 = confx[corr_coo2D(oy+1, ox  )];
  sx11 = confx[corr_coo2D(oy+1, ox+1)];
  sx12 = confx[corr_coo2D(oy+1, ox+2)];
  sx20 = confx[corr_coo2D(oy+2, ox  )];
  sx21 = confx[corr_coo2D(oy+2, ox+1)];
  sx22 = confx[corr_coo2D(oy+2, ox+2)];
  corr[corr_coo2D(ty,tx)] +=  sx00 * cosf(sinshift + fy0*float(oy0) + fx0*float(ox0)) +
			      sx01 * cosf(sinshift + fy0*float(oy0) + fx0*float(ox1)) +
			      sx02 * cosf(sinshift + fy0*float(oy0) + fx0*float(ox2)) +
			      sx10 * cosf(sinshift + fy0*float(oy1) + fx0*float(ox0)) +
			      sx11 * cosf(sinshift + fy0*float(oy1) + fx0*float(ox1)) +
			      sx12 * cosf(sinshift + fy0*float(oy1) + fx0*float(ox2)) +
			      sx20 * cosf(sinshift + fy0*float(oy2) + fx0*float(ox0)) +
			      sx21 * cosf(sinshift + fy0*float(oy2) + fx0*float(ox1)) +
			      sx22 * cosf(sinshift + fy0*float(oy2) + fx0*float(ox2)) ;
  corr[corr_coo2D(ty,tx+1)] += sx00 * cosf(sinshift + fy0*float(oy0) + fx1*float(ox0)) +
			       sx01 * cosf(sinshift + fy0*float(oy0) + fx1*float(ox1)) +
			       sx02 * cosf(sinshift + fy0*float(oy0) + fx1*float(ox2)) +
			       sx10 * cosf(sinshift + fy0*float(oy1) + fx1*float(ox0)) +
			       sx11 * cosf(sinshift + fy0*float(oy1) + fx1*float(ox1)) +
			       sx12 * cosf(sinshift + fy0*float(oy1) + fx1*float(ox2)) +
			       sx20 * cosf(sinshift + fy0*float(oy2) + fx1*float(ox0)) +
			       sx21 * cosf(sinshift + fy0*float(oy2) + fx1*float(ox1)) +
			       sx22 * cosf(sinshift + fy0*float(oy2) + fx1*float(ox2)) ;
  corr[corr_coo2D(ty,tx+2)] += sx00 * cosf(sinshift + fy0*float(oy0) + fx2*float(ox0)) +
			       sx01 * cosf(sinshift + fy0*float(oy0) + fx2*float(ox1)) +
			       sx02 * cosf(sinshift + fy0*float(oy0) + fx2*float(ox2)) +
			       sx10 * cosf(sinshift + fy0*float(oy1) + fx2*float(ox0)) +
			       sx11 * cosf(sinshift + fy0*float(oy1) + fx2*float(ox1)) +
			       sx12 * cosf(sinshift + fy0*float(oy1) + fx2*float(ox2)) +
			       sx20 * cosf(sinshift + fy0*float(oy2) + fx2*float(ox0)) +
			       sx21 * cosf(sinshift + fy0*float(oy2) + fx2*float(ox1)) +
			       sx22 * cosf(sinshift + fy0*float(oy2) + fx2*float(ox2)) ;
  corr[corr_coo2D((ty+1),tx)] += sx00 * cosf(sinshift + fy1*float(oy0) + fx0*float(ox0)) +
			 	 sx01 * cosf(sinshift + fy1*float(oy1) + fx0*float(ox1)) +
			 	 sx02 * cosf(sinshift + fy1*float(oy2) + fx0*float(ox2)) +
			 	 sx10 * cosf(sinshift + fy1*float(oy0) + fx0*float(ox0)) +
			 	 sx11 * cosf(sinshift + fy1*float(oy1) + fx0*float(ox1)) +
			 	 sx12 * cosf(sinshift + fy1*float(oy2) + fx0*float(ox2)) +
			 	 sx20 * cosf(sinshift + fy1*float(oy0) + fx0*float(ox0)) +
			 	 sx21 * cosf(sinshift + fy1*float(oy1) + fx0*float(ox1)) +
			 	 sx22 * cosf(sinshift + fy1*float(oy2) + fx0*float(ox2)) ;
  corr[corr_coo2D((ty+1),tx+1)] += sx00 * cosf(sinshift + fy1*float(oy0) + fx1*float(ox0)) +
				   sx01 * cosf(sinshift + fy1*float(oy1) + fx1*float(ox1)) +
				   sx02 * cosf(sinshift + fy1*float(oy2) + fx1*float(ox2)) +
				   sx10 * cosf(sinshift + fy1*float(oy0) + fx1*float(ox0)) +
				   sx11 * cosf(sinshift + fy1*float(oy1) + fx1*float(ox1)) +
				   sx12 * cosf(sinshift + fy1*float(oy2) + fx1*float(ox2)) +
				   sx20 * cosf(sinshift + fy1*float(oy0) + fx1*float(ox0)) +
				   sx21 * cosf(sinshift + fy1*float(oy1) + fx1*float(ox1)) +
				   sx22 * cosf(sinshift + fy1*float(oy2) + fx1*float(ox2)) ;
  corr[corr_coo2D((ty+1),tx+2)] += sx00 * cosf(sinshift + fy1*float(oy0) + fx2*float(ox0)) +
				   sx01 * cosf(sinshift + fy1*float(oy1) + fx2*float(ox1)) +
				   sx02 * cosf(sinshift + fy1*float(oy2) + fx2*float(ox2)) +
				   sx10 * cosf(sinshift + fy1*float(oy0) + fx2*float(ox0)) +
				   sx11 * cosf(sinshift + fy1*float(oy1) + fx2*float(ox1)) +
				   sx12 * cosf(sinshift + fy1*float(oy2) + fx2*float(ox2)) +
				   sx20 * cosf(sinshift + fy1*float(oy0) + fx2*float(ox0)) +
				   sx21 * cosf(sinshift + fy1*float(oy1) + fx2*float(ox1)) +
				   sx22 * cosf(sinshift + fy1*float(oy2) + fx2*float(ox2)) ;
  corr[corr_coo2D((ty+2),tx)] += sx00 * cosf(sinshift + fy2*float(oy0) + fx0*float(ox0)) +
				 sx01 * cosf(sinshift + fy2*float(oy1) + fx0*float(ox1)) +
				 sx02 * cosf(sinshift + fy2*float(oy2) + fx0*float(ox2)) +
				 sx10 * cosf(sinshift + fy2*float(oy0) + fx0*float(ox0)) +
				 sx11 * cosf(sinshift + fy2*float(oy1) + fx0*float(ox1)) +
				 sx12 * cosf(sinshift + fy2*float(oy2) + fx0*float(ox2)) +
				 sx20 * cosf(sinshift + fy2*float(oy0) + fx0*float(ox0)) +
				 sx21 * cosf(sinshift + fy2*float(oy1) + fx0*float(ox1)) +
				 sx22 * cosf(sinshift + fy2*float(oy2) + fx0*float(ox2)) ;
  corr[corr_coo2D((ty+2),tx+1)] += sx00 * cosf(sinshift + fy2*float(oy0) + fx1*float(ox0)) +
				   sx01 * cosf(sinshift + fy2*float(oy1) + fx1*float(ox1)) +
				   sx02 * cosf(sinshift + fy2*float(oy2) + fx1*float(ox2)) +
				   sx10 * cosf(sinshift + fy2*float(oy0) + fx1*float(ox0)) +
				   sx11 * cosf(sinshift + fy2*float(oy1) + fx1*float(ox1)) +
				   sx12 * cosf(sinshift + fy2*float(oy2) + fx1*float(ox2)) +
				   sx20 * cosf(sinshift + fy2*float(oy0) + fx1*float(ox0)) +
				   sx21 * cosf(sinshift + fy2*float(oy1) + fx1*float(ox1)) +
				   sx22 * cosf(sinshift + fy2*float(oy2) + fx1*float(ox2)) ;
  corr[corr_coo2D((ty+2),tx+2)] += sx00 * cosf(sinshift + fy2*float(oy0) + fx2*float(ox0)) +
				   sx01 * cosf(sinshift + fy2*float(oy1) + fx2*float(ox1)) +
				   sx02 * cosf(sinshift + fy2*float(oy2) + fx2*float(ox2)) +
				   sx10 * cosf(sinshift + fy2*float(oy0) + fx2*float(ox0)) +
				   sx11 * cosf(sinshift + fy2*float(oy1) + fx2*float(ox1)) +
				   sx12 * cosf(sinshift + fy2*float(oy2) + fx2*float(ox2)) +
				   sx20 * cosf(sinshift + fy2*float(oy0) + fx2*float(ox0)) +
				   sx21 * cosf(sinshift + fy2*float(oy1) + fx2*float(ox1)) +
				   sx22 * cosf(sinshift + fy2*float(oy2) + fx2*float(ox2)) ;
  __syncthreads();
}

__global__ void sumcorrTRI(double *DSum_corr, const float *corr, int *DTo){
  //Energy variables
  const int x = threadIdx.x % (corr_BlockSize_x);
  const int y = (threadIdx.x / corr_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
  const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
  const int ty_pt =(DTo[blockIdx.x / corr_BN]) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
  int tx_1 = tx%(corr_SpinSize/2);
  int tx_2 = tx%(corr_SpinSize/2);
  //calculate all the final position first
  DSum_corr[corr_coo2D(ty_pt,tx)] += sqrt(corr[corr_coo2D(ty,tx_2)]*corr[corr_coo2D(ty,tx_2)]+corr[corr_coo2D(ty,tx_1)]*corr[corr_coo2D(ty,tx_1)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D(ty_pt,tx+1)] += sqrt(corr[corr_coo2D(ty,tx_2+1)]*corr[corr_coo2D(ty,tx_2+1)]+corr[corr_coo2D(ty,tx_1+1)]*corr[corr_coo2D(ty,tx_1+1)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D(ty_pt,tx+2)] += sqrt(corr[corr_coo2D(ty,tx_2+2)]*corr[corr_coo2D(ty,tx_2+2)]+corr[corr_coo2D(ty,tx_1+2)]*corr[corr_coo2D(ty,tx_1+2)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty_pt + 1),tx)] += sqrt(corr[corr_coo2D((ty + 1),tx_2)]*corr[corr_coo2D((ty + 1),tx_2)]+corr[corr_coo2D((ty + 1),tx_1)]*corr[corr_coo2D((ty + 1),tx_1)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty_pt + 1),tx+1)] += sqrt(corr[corr_coo2D((ty + 1),tx_2+1)]*corr[corr_coo2D((ty + 1),tx_2+1)]+corr[corr_coo2D((ty + 1),tx_1+1)]*corr[corr_coo2D((ty + 1),tx_1+1)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty_pt + 1),tx+2)] += sqrt(corr[corr_coo2D((ty + 1),tx_2+2)]*corr[corr_coo2D((ty + 1),tx_2+2)]+corr[corr_coo2D((ty + 1),tx_1+2)]*corr[corr_coo2D((ty + 1),tx_1+2)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty_pt + 2),tx)] += sqrt(corr[corr_coo2D((ty + 2),tx_2)]*corr[corr_coo2D((ty + 2),tx_2)]+corr[corr_coo2D((ty + 2),tx_1)]*corr[corr_coo2D((ty + 2),tx_1)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty_pt + 2),tx+1)] += sqrt(corr[corr_coo2D((ty + 2),tx_2+1)]*corr[corr_coo2D((ty + 2),tx_2+1)]+corr[corr_coo2D((ty + 2),tx_1+1)]*corr[corr_coo2D((ty + 2),tx_1+1)])/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty_pt + 2),tx+2)] += sqrt(corr[corr_coo2D((ty + 2),tx_2+2)]*corr[corr_coo2D((ty + 2),tx_2+2)]+corr[corr_coo2D((ty + 2),tx_1+2)]*corr[corr_coo2D((ty + 2),tx_1+2)])/corr_SpinSize/corr_SpinSize;
  __syncthreads();
}

__global__ void getcorrTRI_z(const float *confx, const float *confy, const float *confz, float *corr, int original_i, int original_j){
  /*****************************************************************
    !!!!!!!!!!!!!!! It can be used for square lattice and triangular lattice.
    Set ( original_i, original_j) as our original point.
    for tx_o , ty_o in 2x2 block of (original_i, original_j):
    corr[i - tx_o][j - ty_o] <-  the correlation between  and  (i, j)
    corr[   tx   ][   ty   ]
    use the periodic condition to keep the index positive.
    We need to sum over different (original_i, original_j) to get the correlation.
   *****************************************************************/
  //Energy variables
  const int x = threadIdx.x % (corr_BlockSize_x);
  const int y = (threadIdx.x / corr_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
  const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
  const int ox = original_i;
  const int oy =(blockIdx.x / corr_BN) * corr_SpinSize + original_j;
  //const int txp = tx +1 ;
  //const int typ = ty +1 ;
  //const int ty = 2 * ((blockIdx.x / BN) * SpinSize + ((blockIdx.x % BN) / GridSize_x) * BlockSize_y + y);
  float sz00, sz01, sz02,
        sz10, sz11, sz12,
        sz20, sz21, sz22;
  int fx0, fy0,
      fx1, fy1,
      fx2, fy2,
      fx3, fy3,//from o to f
      fx4, fy4;
  //calculate all the final position first

  fx0 = (tx + original_i) % corr_SpinSize;
  fx1 = (tx + original_i + 1) % corr_SpinSize;
  fx2 = (tx + original_i + 2) % corr_SpinSize;
  fx3 = (tx + original_i + 3) % corr_SpinSize;
  fx4 = (tx + original_i + 4) % corr_SpinSize;

  if((ty % corr_SpinSize + original_j) >= corr_SpinSize)	fy0 = ty + original_j - corr_SpinSize;
  else  fy0 = ty + original_j;
  if((ty % corr_SpinSize + original_j + 1) >= corr_SpinSize)	fy1 = ty + original_j + 1 - corr_SpinSize;
  else  fy1 = ty + original_j + 1;
  if((ty % corr_SpinSize + original_j + 2) >= corr_SpinSize)	fy2 = ty + original_j + 2 - corr_SpinSize;
  else  fy2 = ty + original_j + 2;
  if((ty % corr_SpinSize + original_j + 3) >= corr_SpinSize)	fy3 = ty + original_j + 3 - corr_SpinSize;
  else  fy3 = ty + original_j + 3;
  if((ty % corr_SpinSize + original_j + 4) >= corr_SpinSize)	fy4 = ty + original_j + 4 - corr_SpinSize;
  else  fy4 = ty + original_j + 4;

  //Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.
  sz00 = confz[corr_coo2D(oy,ox)];
  sz01 = confz[corr_coo2D(oy,ox+1)];
  sz02 = confz[corr_coo2D(oy,ox+2)];
  sz10 = confz[corr_coo2D(oy+1,ox)];
  sz11 = confz[corr_coo2D(oy+1,ox+1)];
  sz12 = confz[corr_coo2D(oy+1,ox+2)];
  sz20 = confz[corr_coo2D(oy+2,ox)];
  sz21 = confz[corr_coo2D(oy+2,ox+1)];
  sz22 = confz[corr_coo2D(oy+2,ox+2)];
  corr[corr_coo2D(ty,tx)] += sz00 * confz[corr_coo2D( fy0,fx0)] +
                        sz01 * confz[corr_coo2D( fy0,fx1)] +
                        sz02 * confz[corr_coo2D( fy0,fx2)] +
                        sz10 * confz[corr_coo2D( fy1,fx0)] +
                        sz11 * confz[corr_coo2D( fy1,fx1)] +
                        sz12 * confz[corr_coo2D( fy1,fx2)] +
                        sz20 * confz[corr_coo2D( fy2,fx0)] +
                        sz21 * confz[corr_coo2D( fy2,fx1)] +
                        sz22 * confz[corr_coo2D( fy2,fx2)];
  corr[corr_coo2D(ty,tx+1)] += sz00 * confz[corr_coo2D( fy0,fx1)] +
                          sz01 * confz[corr_coo2D( fy0,fx2)] +
                          sz02 * confz[corr_coo2D( fy0,fx3)] +
                          sz10 * confz[corr_coo2D( fy1,fx1)] +
                          sz11 * confz[corr_coo2D( fy1,fx2)] +
                          sz12 * confz[corr_coo2D( fy1,fx3)] +
                          sz20 * confz[corr_coo2D( fy2,fx1)] +
                          sz21 * confz[corr_coo2D( fy2,fx2)] +
                          sz22 * confz[corr_coo2D( fy2,fx3)];
  corr[corr_coo2D(ty,tx+2)] += sz00 * confz[corr_coo2D( fy0,fx2)] +
                          sz01 * confz[corr_coo2D( fy0,fx3)] +
                          sz02 * confz[corr_coo2D( fy0,fx4)] +
                          sz10 * confz[corr_coo2D( fy1,fx2)] +
                          sz11 * confz[corr_coo2D( fy1,fx3)] +
                          sz12 * confz[corr_coo2D( fy1,fx4)] +
                          sz20 * confz[corr_coo2D( fy2,fx2)] +
                          sz21 * confz[corr_coo2D( fy2,fx3)] +
                          sz22 * confz[corr_coo2D( fy2,fx4)];
  corr[corr_coo2D((ty+1),tx)] += sz00 * confz[corr_coo2D( fy1,fx0)] +
                            sz01 * confz[corr_coo2D( fy1,fx1)] +
                            sz02 * confz[corr_coo2D( fy1,fx2)] +
                            sz10 * confz[corr_coo2D( fy2,fx0)] +
                            sz11 * confz[corr_coo2D( fy2,fx1)] +
                            sz12 * confz[corr_coo2D( fy2,fx2)] +
                            sz20 * confz[corr_coo2D( fy3,fx0)] +
                            sz21 * confz[corr_coo2D( fy3,fx1)] +
                            sz22 * confz[corr_coo2D( fy3,fx2)];
  corr[corr_coo2D((ty+1),tx+1)] += sz00 * confz[corr_coo2D( fy1,fx1)] +
                              sz01 * confz[corr_coo2D( fy1,fx2)] +
                              sz02 * confz[corr_coo2D( fy1,fx3)] +
                              sz10 * confz[corr_coo2D( fy2,fx1)] +
                              sz11 * confz[corr_coo2D( fy2,fx2)] +
                              sz12 * confz[corr_coo2D( fy2,fx3)] +
                              sz20 * confz[corr_coo2D( fy3,fx1)] +
                              sz21 * confz[corr_coo2D( fy3,fx2)] +
                              sz22 * confz[corr_coo2D( fy3,fx3)];
  corr[corr_coo2D((ty+1),tx+2)] += sz00 * confz[corr_coo2D( fy1,fx2)] +
                              sz01 * confz[corr_coo2D( fy1,fx3)] +
                              sz02 * confz[corr_coo2D( fy1,fx4)] +
                              sz10 * confz[corr_coo2D( fy2,fx2)] +
                              sz11 * confz[corr_coo2D( fy2,fx3)] +
                              sz12 * confz[corr_coo2D( fy2,fx4)] +
                              sz20 * confz[corr_coo2D( fy3,fx2)] +
                              sz21 * confz[corr_coo2D( fy3,fx3)] +
                              sz22 * confz[corr_coo2D( fy3,fx4)];
  corr[corr_coo2D((ty+2),tx)] += sz00 * confz[corr_coo2D( fy2,fx0)] +
                            sz01 * confz[corr_coo2D( fy2,fx1)] +
                            sz02 * confz[corr_coo2D( fy2,fx2)] +
                            sz10 * confz[corr_coo2D( fy3,fx0)] +
                            sz11 * confz[corr_coo2D( fy3,fx1)] +
                            sz12 * confz[corr_coo2D( fy3,fx2)] +
                            sz20 * confz[corr_coo2D( fy4,fx0)] +
                            sz21 * confz[corr_coo2D( fy4,fx1)] +
                            sz22 * confz[corr_coo2D( fy4,fx2)];
  corr[corr_coo2D((ty+2),tx+1)] += sz00 * confz[corr_coo2D( fy2,fx1)] +
                              sz01 * confz[corr_coo2D( fy2,fx2)] +
                              sz02 * confz[corr_coo2D( fy2,fx3)] +
                              sz10 * confz[corr_coo2D( fy3,fx1)] +
                              sz11 * confz[corr_coo2D( fy3,fx2)] +
                              sz12 * confz[corr_coo2D( fy3,fx3)] +
                              sz20 * confz[corr_coo2D( fy4,fx1)] +
                              sz21 * confz[corr_coo2D( fy4,fx2)] +
                              sz22 * confz[corr_coo2D( fy4,fx3)];
  corr[corr_coo2D((ty+2),tx+2)] += sz00 * confz[corr_coo2D( fy2,fx2)] +
                              sz01 * confz[corr_coo2D( fy2,fx3)] +
                              sz02 * confz[corr_coo2D( fy2,fx4)] +
                              sz10 * confz[corr_coo2D( fy3,fx2)] +
                              sz11 * confz[corr_coo2D( fy3,fx3)] +
                              sz12 * confz[corr_coo2D( fy3,fx4)] +
                              sz20 * confz[corr_coo2D( fy4,fx2)] +
                              sz21 * confz[corr_coo2D( fy4,fx3)] +
                              sz22 * confz[corr_coo2D( fy4,fx4)];
  __syncthreads();
}

__global__ void avgcorrTRI(double *DSum_corr, double N_corr){
  /*****************************************************************
    Set ( original_i, original_j) as our original point.
    for tx_o , ty_o in 2x2 block of (original_i, original_j):
    corr[i - tx_o][j - ty_o] <-  the correlation between  and  (i, j)
    corr[   tx   ][   ty   ]
    use the periodic condition to keep the index positive.
    We need to sum over different (original_i, original_j) to get the correlation.
   *****************************************************************/
  //Energy variables
  const int x = threadIdx.x % (corr_BlockSize_x);
  const int y = (threadIdx.x / corr_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
  const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
  //calculate all the final position first
  DSum_corr[corr_coo2D(ty,tx)] = DSum_corr[corr_coo2D(ty,tx)]/N_corr;
  DSum_corr[corr_coo2D(ty,tx+1)] = DSum_corr[corr_coo2D(ty,tx+1)]/N_corr;
  DSum_corr[corr_coo2D(ty,tx+2)] = DSum_corr[corr_coo2D(ty,tx+2)]/N_corr;
  DSum_corr[corr_coo2D((ty + 1),tx)] = DSum_corr[corr_coo2D((ty + 1),tx)]/N_corr;
  DSum_corr[corr_coo2D((ty + 1),tx+1)] = DSum_corr[corr_coo2D((ty + 1),tx+1)]/N_corr;
  DSum_corr[corr_coo2D((ty + 1),tx+2)] = DSum_corr[corr_coo2D((ty + 1),tx+2)]/N_corr;
  DSum_corr[corr_coo2D((ty + 2),tx)] = DSum_corr[corr_coo2D((ty + 2),tx)]/N_corr;
  DSum_corr[corr_coo2D((ty + 2),tx+1)] = DSum_corr[corr_coo2D((ty + 2),tx+1)]/N_corr;
  DSum_corr[corr_coo2D((ty + 2),tx+2)] = DSum_corr[corr_coo2D((ty + 2),tx+2)]/N_corr;
  __syncthreads();
}
#endif
