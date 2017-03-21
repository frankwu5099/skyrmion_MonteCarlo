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
__global__ void getcorrTRI(const float *confx, const float *confy, const float *confz, float *corr, int original_i, int original_j){
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
  float sx00, sy00, sz00, sx01, sy01, sz01, sx02, sy02, sz02,
        sx10, sy10, sz10, sx11, sy11, sz11, sx12, sy12, sz12,
        sx20, sy20, sz20, sx21, sy21, sz21, sx22, sy22, sz22;
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
  sy00 = confy[corr_coo2D(oy,ox)];
  sz00 = confz[corr_coo2D(oy,ox)];
  sx01 = confx[corr_coo2D(oy,ox+1)];
  sy01 = confy[corr_coo2D(oy,ox+1)];
  sz01 = confz[corr_coo2D(oy,ox+1)];
  sx02 = confx[corr_coo2D(oy,ox+2)];
  sy02 = confy[corr_coo2D(oy,ox+2)];
  sz02 = confz[corr_coo2D(oy,ox+2)];
  sx10 = confx[corr_coo2D(oy+1,ox)];
  sy10 = confy[corr_coo2D(oy+1,ox)];
  sz10 = confz[corr_coo2D(oy+1,ox)];
  sx11 = confx[corr_coo2D(oy+1,ox+1)];
  sy11 = confy[corr_coo2D(oy+1,ox+1)];
  sz11 = confz[corr_coo2D(oy+1,ox+1)];
  sx12 = confx[corr_coo2D(oy+1,ox+2)];
  sy12 = confy[corr_coo2D(oy+1,ox+2)];
  sz12 = confz[corr_coo2D(oy+1,ox+2)];
  sx20 = confx[corr_coo2D(oy+2,ox)];
  sy20 = confy[corr_coo2D(oy+2,ox)];
  sz20 = confz[corr_coo2D(oy+2,ox)];
  sx21 = confx[corr_coo2D(oy+2,ox+1)];
  sy21 = confy[corr_coo2D(oy+2,ox+1)];
  sz21 = confz[corr_coo2D(oy+2,ox+1)];
  sx22 = confx[corr_coo2D(oy+2,ox+2)];
  sy22 = confy[corr_coo2D(oy+2,ox+2)];
  sz22 = confz[corr_coo2D(oy+2,ox+2)];
  corr[corr_coo2D(ty,tx)] += sx00 * confx[corr_coo2D( fy0,fx0)] + sy00 * confy[corr_coo2D( fy0,fx0)] + sz00 * confz[corr_coo2D( fy0,fx0)] +
                        sx01 * confx[corr_coo2D( fy0,fx1)] + sy01 * confy[corr_coo2D( fy0,fx1)] + sz01 * confz[corr_coo2D( fy0,fx1)] +
                        sx02 * confx[corr_coo2D( fy0,fx2)] + sy02 * confy[corr_coo2D( fy0,fx2)] + sz02 * confz[corr_coo2D( fy0,fx2)] +
                        sx10 * confx[corr_coo2D( fy1,fx0)] + sy10 * confy[corr_coo2D( fy1,fx0)] + sz10 * confz[corr_coo2D( fy1,fx0)] +
                        sx11 * confx[corr_coo2D( fy1,fx1)] + sy11 * confy[corr_coo2D( fy1,fx1)] + sz11 * confz[corr_coo2D( fy1,fx1)] +
                        sx12 * confx[corr_coo2D( fy1,fx2)] + sy12 * confy[corr_coo2D( fy1,fx2)] + sz12 * confz[corr_coo2D( fy1,fx2)] +
                        sx20 * confx[corr_coo2D( fy2,fx0)] + sy20 * confy[corr_coo2D( fy2,fx0)] + sz20 * confz[corr_coo2D( fy2,fx0)] +
                        sx21 * confx[corr_coo2D( fy2,fx1)] + sy21 * confy[corr_coo2D( fy2,fx1)] + sz21 * confz[corr_coo2D( fy2,fx1)] +
                        sx22 * confx[corr_coo2D( fy2,fx2)] + sy22 * confy[corr_coo2D( fy2,fx2)] + sz22 * confz[corr_coo2D( fy2,fx2)];
  corr[corr_coo2D(ty,tx+1)] += sx00 * confx[corr_coo2D( fy0,fx1)] + sy00 * confy[corr_coo2D( fy0,fx1)] + sz00 * confz[corr_coo2D( fy0,fx1)] +
                          sx01 * confx[corr_coo2D( fy0,fx2)] + sy01 * confy[corr_coo2D( fy0,fx2)] + sz01 * confz[corr_coo2D( fy0,fx2)] +
                          sx02 * confx[corr_coo2D( fy0,fx3)] + sy02 * confy[corr_coo2D( fy0,fx3)] + sz02 * confz[corr_coo2D( fy0,fx3)] +
                          sx10 * confx[corr_coo2D( fy1,fx1)] + sy10 * confy[corr_coo2D( fy1,fx1)] + sz10 * confz[corr_coo2D( fy1,fx1)] +
                          sx11 * confx[corr_coo2D( fy1,fx2)] + sy11 * confy[corr_coo2D( fy1,fx2)] + sz11 * confz[corr_coo2D( fy1,fx2)] +
                          sx12 * confx[corr_coo2D( fy1,fx3)] + sy12 * confy[corr_coo2D( fy1,fx3)] + sz12 * confz[corr_coo2D( fy1,fx3)] +
                          sx20 * confx[corr_coo2D( fy2,fx1)] + sy20 * confy[corr_coo2D( fy2,fx1)] + sz20 * confz[corr_coo2D( fy2,fx1)] +
                          sx21 * confx[corr_coo2D( fy2,fx2)] + sy21 * confy[corr_coo2D( fy2,fx2)] + sz21 * confz[corr_coo2D( fy2,fx2)] +
                          sx22 * confx[corr_coo2D( fy2,fx3)] + sy22 * confy[corr_coo2D( fy2,fx3)] + sz22 * confz[corr_coo2D( fy2,fx3)];
  corr[corr_coo2D(ty,tx+2)] += sx00 * confx[corr_coo2D( fy0,fx2)] + sy00 * confy[corr_coo2D( fy0,fx2)] + sz00 * confz[corr_coo2D( fy0,fx2)] +
                          sx01 * confx[corr_coo2D( fy0,fx3)] + sy01 * confy[corr_coo2D( fy0,fx3)] + sz01 * confz[corr_coo2D( fy0,fx3)] +
                          sx02 * confx[corr_coo2D( fy0,fx4)] + sy02 * confy[corr_coo2D( fy0,fx4)] + sz02 * confz[corr_coo2D( fy0,fx4)] +
                          sx10 * confx[corr_coo2D( fy1,fx2)] + sy10 * confy[corr_coo2D( fy1,fx2)] + sz10 * confz[corr_coo2D( fy1,fx2)] +
                          sx11 * confx[corr_coo2D( fy1,fx3)] + sy11 * confy[corr_coo2D( fy1,fx3)] + sz11 * confz[corr_coo2D( fy1,fx3)] +
                          sx12 * confx[corr_coo2D( fy1,fx4)] + sy12 * confy[corr_coo2D( fy1,fx4)] + sz12 * confz[corr_coo2D( fy1,fx4)] +
                          sx20 * confx[corr_coo2D( fy2,fx2)] + sy20 * confy[corr_coo2D( fy2,fx2)] + sz20 * confz[corr_coo2D( fy2,fx2)] +
                          sx21 * confx[corr_coo2D( fy2,fx3)] + sy21 * confy[corr_coo2D( fy2,fx3)] + sz21 * confz[corr_coo2D( fy2,fx3)] +
                          sx22 * confx[corr_coo2D( fy2,fx4)] + sy22 * confy[corr_coo2D( fy2,fx4)] + sz22 * confz[corr_coo2D( fy2,fx4)];
  corr[corr_coo2D((ty+1),tx)] += sx00 * confx[corr_coo2D( fy1,fx0)] + sy00 * confy[corr_coo2D( fy1,fx0)] + sz00 * confz[corr_coo2D( fy1,fx0)] +
                            sx01 * confx[corr_coo2D( fy1,fx1)] + sy01 * confy[corr_coo2D( fy1,fx1)] + sz01 * confz[corr_coo2D( fy1,fx1)] +
                            sx02 * confx[corr_coo2D( fy1,fx2)] + sy02 * confy[corr_coo2D( fy1,fx2)] + sz02 * confz[corr_coo2D( fy1,fx2)] +
                            sx10 * confx[corr_coo2D( fy2,fx0)] + sy10 * confy[corr_coo2D( fy2,fx0)] + sz10 * confz[corr_coo2D( fy2,fx0)] +
                            sx11 * confx[corr_coo2D( fy2,fx1)] + sy11 * confy[corr_coo2D( fy2,fx1)] + sz11 * confz[corr_coo2D( fy2,fx1)] +
                            sx12 * confx[corr_coo2D( fy2,fx2)] + sy12 * confy[corr_coo2D( fy2,fx2)] + sz12 * confz[corr_coo2D( fy2,fx2)] +
                            sx20 * confx[corr_coo2D( fy3,fx0)] + sy20 * confy[corr_coo2D( fy3,fx0)] + sz20 * confz[corr_coo2D( fy3,fx0)] +
                            sx21 * confx[corr_coo2D( fy3,fx1)] + sy21 * confy[corr_coo2D( fy3,fx1)] + sz21 * confz[corr_coo2D( fy3,fx1)] +
                            sx22 * confx[corr_coo2D( fy3,fx2)] + sy22 * confy[corr_coo2D( fy3,fx2)] + sz22 * confz[corr_coo2D( fy3,fx2)];
  corr[corr_coo2D((ty+1),tx+1)] += sx00 * confx[corr_coo2D( fy1,fx1)] + sy00 * confy[corr_coo2D( fy1,fx1)] + sz00 * confz[corr_coo2D( fy1,fx1)] +
                              sx01 * confx[corr_coo2D( fy1,fx2)] + sy01 * confy[corr_coo2D( fy1,fx2)] + sz01 * confz[corr_coo2D( fy1,fx2)] +
                              sx02 * confx[corr_coo2D( fy1,fx3)] + sy02 * confy[corr_coo2D( fy1,fx3)] + sz02 * confz[corr_coo2D( fy1,fx3)] +
                              sx10 * confx[corr_coo2D( fy2,fx1)] + sy10 * confy[corr_coo2D( fy2,fx1)] + sz10 * confz[corr_coo2D( fy2,fx1)] +
                              sx11 * confx[corr_coo2D( fy2,fx2)] + sy11 * confy[corr_coo2D( fy2,fx2)] + sz11 * confz[corr_coo2D( fy2,fx2)] +
                              sx12 * confx[corr_coo2D( fy2,fx3)] + sy12 * confy[corr_coo2D( fy2,fx3)] + sz12 * confz[corr_coo2D( fy2,fx3)] +
                              sx20 * confx[corr_coo2D( fy3,fx1)] + sy20 * confy[corr_coo2D( fy3,fx1)] + sz20 * confz[corr_coo2D( fy3,fx1)] +
                              sx21 * confx[corr_coo2D( fy3,fx2)] + sy21 * confy[corr_coo2D( fy3,fx2)] + sz21 * confz[corr_coo2D( fy3,fx2)] +
                              sx22 * confx[corr_coo2D( fy3,fx3)] + sy22 * confy[corr_coo2D( fy3,fx3)] + sz22 * confz[corr_coo2D( fy3,fx3)];
  corr[corr_coo2D((ty+1),tx+2)] += sx00 * confx[corr_coo2D( fy1,fx2)] + sy00 * confy[corr_coo2D( fy1,fx2)] + sz00 * confz[corr_coo2D( fy1,fx2)] +
                              sx01 * confx[corr_coo2D( fy1,fx3)] + sy01 * confy[corr_coo2D( fy1,fx3)] + sz01 * confz[corr_coo2D( fy1,fx3)] +
                              sx02 * confx[corr_coo2D( fy1,fx4)] + sy02 * confy[corr_coo2D( fy1,fx4)] + sz02 * confz[corr_coo2D( fy1,fx4)] +
                              sx10 * confx[corr_coo2D( fy2,fx2)] + sy10 * confy[corr_coo2D( fy2,fx2)] + sz10 * confz[corr_coo2D( fy2,fx2)] +
                              sx11 * confx[corr_coo2D( fy2,fx3)] + sy11 * confy[corr_coo2D( fy2,fx3)] + sz11 * confz[corr_coo2D( fy2,fx3)] +
                              sx12 * confx[corr_coo2D( fy2,fx4)] + sy12 * confy[corr_coo2D( fy2,fx4)] + sz12 * confz[corr_coo2D( fy2,fx4)] +
                              sx20 * confx[corr_coo2D( fy3,fx2)] + sy20 * confy[corr_coo2D( fy3,fx2)] + sz20 * confz[corr_coo2D( fy3,fx2)] +
                              sx21 * confx[corr_coo2D( fy3,fx3)] + sy21 * confy[corr_coo2D( fy3,fx3)] + sz21 * confz[corr_coo2D( fy3,fx3)] +
                              sx22 * confx[corr_coo2D( fy3,fx4)] + sy22 * confy[corr_coo2D( fy3,fx4)] + sz22 * confz[corr_coo2D( fy3,fx4)];
  corr[corr_coo2D((ty+2),tx)] += sx00 * confx[corr_coo2D( fy2,fx0)] + sy00 * confy[corr_coo2D( fy2,fx0)] + sz00 * confz[corr_coo2D( fy2,fx0)] +
                            sx01 * confx[corr_coo2D( fy2,fx1)] + sy01 * confy[corr_coo2D( fy2,fx1)] + sz01 * confz[corr_coo2D( fy2,fx1)] +
                            sx02 * confx[corr_coo2D( fy2,fx2)] + sy02 * confy[corr_coo2D( fy2,fx2)] + sz02 * confz[corr_coo2D( fy2,fx2)] +
                            sx10 * confx[corr_coo2D( fy3,fx0)] + sy10 * confy[corr_coo2D( fy3,fx0)] + sz10 * confz[corr_coo2D( fy3,fx0)] +
                            sx11 * confx[corr_coo2D( fy3,fx1)] + sy11 * confy[corr_coo2D( fy3,fx1)] + sz11 * confz[corr_coo2D( fy3,fx1)] +
                            sx12 * confx[corr_coo2D( fy3,fx2)] + sy12 * confy[corr_coo2D( fy3,fx2)] + sz12 * confz[corr_coo2D( fy3,fx2)] +
                            sx20 * confx[corr_coo2D( fy4,fx0)] + sy20 * confy[corr_coo2D( fy4,fx0)] + sz20 * confz[corr_coo2D( fy4,fx0)] +
                            sx21 * confx[corr_coo2D( fy4,fx1)] + sy21 * confy[corr_coo2D( fy4,fx1)] + sz21 * confz[corr_coo2D( fy4,fx1)] +
                            sx22 * confx[corr_coo2D( fy4,fx2)] + sy22 * confy[corr_coo2D( fy4,fx2)] + sz22 * confz[corr_coo2D( fy4,fx2)];
  corr[corr_coo2D((ty+2),tx+1)] += sx00 * confx[corr_coo2D( fy2,fx1)] + sy00 * confy[corr_coo2D( fy2,fx1)] + sz00 * confz[corr_coo2D( fy2,fx1)] +
                              sx01 * confx[corr_coo2D( fy2,fx2)] + sy01 * confy[corr_coo2D( fy2,fx2)] + sz01 * confz[corr_coo2D( fy2,fx2)] +
                              sx02 * confx[corr_coo2D( fy2,fx3)] + sy02 * confy[corr_coo2D( fy2,fx3)] + sz02 * confz[corr_coo2D( fy2,fx3)] +
                              sx10 * confx[corr_coo2D( fy3,fx1)] + sy10 * confy[corr_coo2D( fy3,fx1)] + sz10 * confz[corr_coo2D( fy3,fx1)] +
                              sx11 * confx[corr_coo2D( fy3,fx2)] + sy11 * confy[corr_coo2D( fy3,fx2)] + sz11 * confz[corr_coo2D( fy3,fx2)] +
                              sx12 * confx[corr_coo2D( fy3,fx3)] + sy12 * confy[corr_coo2D( fy3,fx3)] + sz12 * confz[corr_coo2D( fy3,fx3)] +
                              sx20 * confx[corr_coo2D( fy4,fx1)] + sy20 * confy[corr_coo2D( fy4,fx1)] + sz20 * confz[corr_coo2D( fy4,fx1)] +
                              sx21 * confx[corr_coo2D( fy4,fx2)] + sy21 * confy[corr_coo2D( fy4,fx2)] + sz21 * confz[corr_coo2D( fy4,fx2)] +
                              sx22 * confx[corr_coo2D( fy4,fx3)] + sy22 * confy[corr_coo2D( fy4,fx3)] + sz22 * confz[corr_coo2D( fy4,fx3)];
  corr[corr_coo2D((ty+2),tx+2)] += sx00 * confx[corr_coo2D( fy2,fx2)] + sy00 * confy[corr_coo2D( fy2,fx2)] + sz00 * confz[corr_coo2D( fy2,fx2)] +
                              sx01 * confx[corr_coo2D( fy2,fx3)] + sy01 * confy[corr_coo2D( fy2,fx3)] + sz01 * confz[corr_coo2D( fy2,fx3)] +
                              sx02 * confx[corr_coo2D( fy2,fx4)] + sy02 * confy[corr_coo2D( fy2,fx4)] + sz02 * confz[corr_coo2D( fy2,fx4)] +
                              sx10 * confx[corr_coo2D( fy3,fx2)] + sy10 * confy[corr_coo2D( fy3,fx2)] + sz10 * confz[corr_coo2D( fy3,fx2)] +
                              sx11 * confx[corr_coo2D( fy3,fx3)] + sy11 * confy[corr_coo2D( fy3,fx3)] + sz11 * confz[corr_coo2D( fy3,fx3)] +
                              sx12 * confx[corr_coo2D( fy3,fx4)] + sy12 * confy[corr_coo2D( fy3,fx4)] + sz12 * confz[corr_coo2D( fy3,fx4)] +
                              sx20 * confx[corr_coo2D( fy4,fx2)] + sy20 * confy[corr_coo2D( fy4,fx2)] + sz20 * confz[corr_coo2D( fy4,fx2)] +
                              sx21 * confx[corr_coo2D( fy4,fx3)] + sy21 * confy[corr_coo2D( fy4,fx3)] + sz21 * confz[corr_coo2D( fy4,fx3)] +
                              sx22 * confx[corr_coo2D( fy4,fx4)] + sy22 * confy[corr_coo2D( fy4,fx4)] + sz22 * confz[corr_coo2D( fy4,fx4)];
  __syncthreads();
}

__global__ void sumcorrTRI(double *DSum_corr, const float *corr, int *DTo){
  //Energy variables
  const int x = threadIdx.x % (corr_BlockSize_x);
  const int y = (threadIdx.x / corr_BlockSize_x);
  const int tx = 3 * (((blockIdx.x % corr_BN) % corr_GridSize_x) * corr_BlockSize_x + x);
  const int ty =(blockIdx.x / corr_BN) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
	//const int ty_pt =(DTo[blockIdx.x / corr_BN]) * corr_SpinSize +  3 * ((((blockIdx.x % corr_BN) / corr_GridSize_x) % corr_GridSize_y) * corr_BlockSize_y + y);
  //calculate all the final position first
  DSum_corr[corr_coo2D(ty,tx)] += corr[corr_coo2D(ty,tx)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D(ty,tx+1)] += corr[corr_coo2D(ty,tx+1)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D(ty,tx+2)] += corr[corr_coo2D(ty,tx+2)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty + 1),tx)] += corr[corr_coo2D((ty + 1),tx)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty + 1),tx+1)] += corr[corr_coo2D((ty + 1),tx+1)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty + 1),tx+2)] += corr[corr_coo2D((ty + 1),tx+2)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty + 2),tx)] += corr[corr_coo2D((ty + 2),tx)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty + 2),tx+1)] += corr[corr_coo2D((ty + 2),tx+1)]/corr_SpinSize/corr_SpinSize;
  DSum_corr[corr_coo2D((ty + 2),tx+2)] += corr[corr_coo2D((ty + 2),tx+2)]/corr_SpinSize/corr_SpinSize;
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
