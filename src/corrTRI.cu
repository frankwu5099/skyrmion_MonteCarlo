#include "measurements.cuh"
#ifdef TRI
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
  const int x = threadIdx.x % (BlockSize_x);
  const int y = (threadIdx.x / BlockSize_x);
  const int tx = 3 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
  const int ty =(blockIdx.x / BN) * SpinSize +  3 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
  const int ox = original_i;
  const int oy =(blockIdx.x / BN) * SpinSize + original_j;
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

  fx0 = (tx + original_i) % SpinSize;
  fx1 = (tx + original_i + 1) % SpinSize;
  fx2 = (tx + original_i + 2) % SpinSize;
  fx3 = (tx + original_i + 3) % SpinSize;
  fx4 = (tx + original_i + 4) % SpinSize;

  if((ty % SpinSize + original_j) >= SpinSize)	fy0 = ty + original_j - SpinSize;
  else  fy0 = ty + original_j;
  if((ty % SpinSize + original_j + 1) >= SpinSize)	fy1 = ty + original_j + 1 - SpinSize;
  else  fy1 = ty + original_j + 1;
  if((ty % SpinSize + original_j + 2) >= SpinSize)	fy2 = ty + original_j + 2 - SpinSize;
  else  fy2 = ty + original_j + 2;
  if((ty % SpinSize + original_j + 3) >= SpinSize)	fy3 = ty + original_j + 3 - SpinSize;
  else  fy3 = ty + original_j + 3;
  if((ty % SpinSize + original_j + 4) >= SpinSize)	fy4 = ty + original_j + 4 - SpinSize;
  else  fy4 = ty + original_j + 4;

  //Calculate the two pair-energy of each spin on the thread square step by step and store the summing energy of each thread square in sD.
  sx00 = confx[coo2D(oy,ox)];
  sy00 = confy[coo2D(oy,ox)];
  sz00 = confz[coo2D(oy,ox)];
  sx01 = confx[coo2D(oy,ox+1)];
  sy01 = confy[coo2D(oy,ox+1)];
  sz01 = confz[coo2D(oy,ox+1)];
  sx02 = confx[coo2D(oy,ox+2)];
  sy02 = confy[coo2D(oy,ox+2)];
  sz02 = confz[coo2D(oy,ox+2)];
  sx10 = confx[coo2D(oy+1,ox)];
  sy10 = confy[coo2D(oy+1,ox)];
  sz10 = confz[coo2D(oy+1,ox)];
  sx11 = confx[coo2D(oy+1,ox+1)];
  sy11 = confy[coo2D(oy+1,ox+1)];
  sz11 = confz[coo2D(oy+1,ox+1)];
  sx12 = confx[coo2D(oy+1,ox+2)];
  sy12 = confy[coo2D(oy+1,ox+2)];
  sz12 = confz[coo2D(oy+1,ox+2)];
  sx20 = confx[coo2D(oy+2,ox)];
  sy20 = confy[coo2D(oy+2,ox)];
  sz20 = confz[coo2D(oy+2,ox)];
  sx21 = confx[coo2D(oy+2,ox+1)];
  sy21 = confy[coo2D(oy+2,ox+1)];
  sz21 = confz[coo2D(oy+2,ox+1)];
  sx22 = confx[coo2D(oy+2,ox+2)];
  sy22 = confy[coo2D(oy+2,ox+2)];
  sz22 = confz[coo2D(oy+2,ox+2)];
  corr[coo2D(ty,tx)] += sx00 * confx[coo2D( fy0,fx0)] + sy00 * confy[coo2D( fy0,fx0)] + sz00 * confz[coo2D( fy0,fx0)] +
                        sx01 * confx[coo2D( fy0,fx1)] + sy01 * confy[coo2D( fy0,fx1)] + sz01 * confz[coo2D( fy0,fx1)] +
                        sx02 * confx[coo2D( fy0,fx2)] + sy02 * confy[coo2D( fy0,fx2)] + sz02 * confz[coo2D( fy0,fx2)] +
                        sx10 * confx[coo2D( fy1,fx0)] + sy10 * confy[coo2D( fy1,fx0)] + sz10 * confz[coo2D( fy1,fx0)] +
                        sx11 * confx[coo2D( fy1,fx1)] + sy11 * confy[coo2D( fy1,fx1)] + sz11 * confz[coo2D( fy1,fx1)] +
                        sx12 * confx[coo2D( fy1,fx2)] + sy12 * confy[coo2D( fy1,fx2)] + sz12 * confz[coo2D( fy1,fx2)] +
                        sx20 * confx[coo2D( fy2,fx0)] + sy20 * confy[coo2D( fy2,fx0)] + sz20 * confz[coo2D( fy2,fx0)] +
                        sx21 * confx[coo2D( fy2,fx1)] + sy21 * confy[coo2D( fy2,fx1)] + sz21 * confz[coo2D( fy2,fx1)] +
                        sx22 * confx[coo2D( fy2,fx2)] + sy22 * confy[coo2D( fy2,fx2)] + sz22 * confz[coo2D( fy2,fx2)];
  corr[coo2D(ty,tx+1)] += sx00 * confx[coo2D( fy0,fx1)] + sy00 * confy[coo2D( fy0,fx1)] + sz00 * confz[coo2D( fy0,fx1)] +
                          sx01 * confx[coo2D( fy0,fx2)] + sy01 * confy[coo2D( fy0,fx2)] + sz01 * confz[coo2D( fy0,fx2)] +
                          sx02 * confx[coo2D( fy0,fx3)] + sy02 * confy[coo2D( fy0,fx3)] + sz02 * confz[coo2D( fy0,fx3)] +
                          sx10 * confx[coo2D( fy1,fx1)] + sy10 * confy[coo2D( fy1,fx1)] + sz10 * confz[coo2D( fy1,fx1)] +
                          sx11 * confx[coo2D( fy1,fx2)] + sy11 * confy[coo2D( fy1,fx2)] + sz11 * confz[coo2D( fy1,fx2)] +
                          sx12 * confx[coo2D( fy1,fx3)] + sy12 * confy[coo2D( fy1,fx3)] + sz12 * confz[coo2D( fy1,fx3)] +
                          sx20 * confx[coo2D( fy2,fx1)] + sy20 * confy[coo2D( fy2,fx1)] + sz20 * confz[coo2D( fy2,fx1)] +
                          sx21 * confx[coo2D( fy2,fx2)] + sy21 * confy[coo2D( fy2,fx2)] + sz21 * confz[coo2D( fy2,fx2)] +
                          sx22 * confx[coo2D( fy2,fx3)] + sy22 * confy[coo2D( fy2,fx3)] + sz22 * confz[coo2D( fy2,fx3)];
  corr[coo2D(ty,tx+2)] += sx00 * confx[coo2D( fy0,fx2)] + sy00 * confy[coo2D( fy0,fx2)] + sz00 * confz[coo2D( fy0,fx2)] +
                          sx01 * confx[coo2D( fy0,fx3)] + sy01 * confy[coo2D( fy0,fx3)] + sz01 * confz[coo2D( fy0,fx3)] +
                          sx02 * confx[coo2D( fy0,fx4)] + sy02 * confy[coo2D( fy0,fx4)] + sz02 * confz[coo2D( fy0,fx4)] +
                          sx10 * confx[coo2D( fy1,fx2)] + sy10 * confy[coo2D( fy1,fx2)] + sz10 * confz[coo2D( fy1,fx2)] +
                          sx11 * confx[coo2D( fy1,fx3)] + sy11 * confy[coo2D( fy1,fx3)] + sz11 * confz[coo2D( fy1,fx3)] +
                          sx12 * confx[coo2D( fy1,fx4)] + sy12 * confy[coo2D( fy1,fx4)] + sz12 * confz[coo2D( fy1,fx4)] +
                          sx20 * confx[coo2D( fy2,fx2)] + sy20 * confy[coo2D( fy2,fx2)] + sz20 * confz[coo2D( fy2,fx2)] +
                          sx21 * confx[coo2D( fy2,fx3)] + sy21 * confy[coo2D( fy2,fx3)] + sz21 * confz[coo2D( fy2,fx3)] +
                          sx22 * confx[coo2D( fy2,fx4)] + sy22 * confy[coo2D( fy2,fx4)] + sz22 * confz[coo2D( fy2,fx4)];
  corr[coo2D((ty+1),tx)] += sx00 * confx[coo2D( fy1,fx0)] + sy00 * confy[coo2D( fy1,fx0)] + sz00 * confz[coo2D( fy1,fx0)] +
                            sx01 * confx[coo2D( fy1,fx1)] + sy01 * confy[coo2D( fy1,fx1)] + sz01 * confz[coo2D( fy1,fx1)] +
                            sx02 * confx[coo2D( fy1,fx2)] + sy02 * confy[coo2D( fy1,fx2)] + sz02 * confz[coo2D( fy1,fx2)] +
                            sx10 * confx[coo2D( fy2,fx0)] + sy10 * confy[coo2D( fy2,fx0)] + sz10 * confz[coo2D( fy2,fx0)] +
                            sx11 * confx[coo2D( fy2,fx1)] + sy11 * confy[coo2D( fy2,fx1)] + sz11 * confz[coo2D( fy2,fx1)] +
                            sx12 * confx[coo2D( fy2,fx2)] + sy12 * confy[coo2D( fy2,fx2)] + sz12 * confz[coo2D( fy2,fx2)] +
                            sx20 * confx[coo2D( fy3,fx0)] + sy20 * confy[coo2D( fy3,fx0)] + sz20 * confz[coo2D( fy3,fx0)] +
                            sx21 * confx[coo2D( fy3,fx1)] + sy21 * confy[coo2D( fy3,fx1)] + sz21 * confz[coo2D( fy3,fx1)] +
                            sx22 * confx[coo2D( fy3,fx2)] + sy22 * confy[coo2D( fy3,fx2)] + sz22 * confz[coo2D( fy3,fx2)];
  corr[coo2D((ty+1),tx+1)] += sx00 * confx[coo2D( fy1,fx1)] + sy00 * confy[coo2D( fy1,fx1)] + sz00 * confz[coo2D( fy1,fx1)] +
                              sx01 * confx[coo2D( fy1,fx2)] + sy01 * confy[coo2D( fy1,fx2)] + sz01 * confz[coo2D( fy1,fx2)] +
                              sx02 * confx[coo2D( fy1,fx3)] + sy02 * confy[coo2D( fy1,fx3)] + sz02 * confz[coo2D( fy1,fx3)] +
                              sx10 * confx[coo2D( fy2,fx1)] + sy10 * confy[coo2D( fy2,fx1)] + sz10 * confz[coo2D( fy2,fx1)] +
                              sx11 * confx[coo2D( fy2,fx2)] + sy11 * confy[coo2D( fy2,fx2)] + sz11 * confz[coo2D( fy2,fx2)] +
                              sx12 * confx[coo2D( fy2,fx3)] + sy12 * confy[coo2D( fy2,fx3)] + sz12 * confz[coo2D( fy2,fx3)] +
                              sx20 * confx[coo2D( fy3,fx1)] + sy20 * confy[coo2D( fy3,fx1)] + sz20 * confz[coo2D( fy3,fx1)] +
                              sx21 * confx[coo2D( fy3,fx2)] + sy21 * confy[coo2D( fy3,fx2)] + sz21 * confz[coo2D( fy3,fx2)] +
                              sx22 * confx[coo2D( fy3,fx3)] + sy22 * confy[coo2D( fy3,fx3)] + sz22 * confz[coo2D( fy3,fx3)];
  corr[coo2D((ty+1),tx+2)] += sx00 * confx[coo2D( fy1,fx2)] + sy00 * confy[coo2D( fy1,fx2)] + sz00 * confz[coo2D( fy1,fx2)] +
                              sx01 * confx[coo2D( fy1,fx3)] + sy01 * confy[coo2D( fy1,fx3)] + sz01 * confz[coo2D( fy1,fx3)] +
                              sx02 * confx[coo2D( fy1,fx4)] + sy02 * confy[coo2D( fy1,fx4)] + sz02 * confz[coo2D( fy1,fx4)] +
                              sx10 * confx[coo2D( fy2,fx2)] + sy10 * confy[coo2D( fy2,fx2)] + sz10 * confz[coo2D( fy2,fx2)] +
                              sx11 * confx[coo2D( fy2,fx3)] + sy11 * confy[coo2D( fy2,fx3)] + sz11 * confz[coo2D( fy2,fx3)] +
                              sx12 * confx[coo2D( fy2,fx4)] + sy12 * confy[coo2D( fy2,fx4)] + sz12 * confz[coo2D( fy2,fx4)] +
                              sx20 * confx[coo2D( fy3,fx2)] + sy20 * confy[coo2D( fy3,fx2)] + sz20 * confz[coo2D( fy3,fx2)] +
                              sx21 * confx[coo2D( fy3,fx3)] + sy21 * confy[coo2D( fy3,fx3)] + sz21 * confz[coo2D( fy3,fx3)] +
                              sx22 * confx[coo2D( fy3,fx4)] + sy22 * confy[coo2D( fy3,fx4)] + sz22 * confz[coo2D( fy3,fx4)];
  corr[coo2D((ty+2),tx)] += sx00 * confx[coo2D( fy2,fx0)] + sy00 * confy[coo2D( fy2,fx0)] + sz00 * confz[coo2D( fy2,fx0)] +
                            sx01 * confx[coo2D( fy2,fx1)] + sy01 * confy[coo2D( fy2,fx1)] + sz01 * confz[coo2D( fy2,fx1)] +
                            sx02 * confx[coo2D( fy2,fx2)] + sy02 * confy[coo2D( fy2,fx2)] + sz02 * confz[coo2D( fy2,fx2)] +
                            sx10 * confx[coo2D( fy3,fx0)] + sy10 * confy[coo2D( fy3,fx0)] + sz10 * confz[coo2D( fy3,fx0)] +
                            sx11 * confx[coo2D( fy3,fx1)] + sy11 * confy[coo2D( fy3,fx1)] + sz11 * confz[coo2D( fy3,fx1)] +
                            sx12 * confx[coo2D( fy3,fx2)] + sy12 * confy[coo2D( fy3,fx2)] + sz12 * confz[coo2D( fy3,fx2)] +
                            sx20 * confx[coo2D( fy4,fx0)] + sy20 * confy[coo2D( fy4,fx0)] + sz20 * confz[coo2D( fy4,fx0)] +
                            sx21 * confx[coo2D( fy4,fx1)] + sy21 * confy[coo2D( fy4,fx1)] + sz21 * confz[coo2D( fy4,fx1)] +
                            sx22 * confx[coo2D( fy4,fx2)] + sy22 * confy[coo2D( fy4,fx2)] + sz22 * confz[coo2D( fy4,fx2)];
  corr[coo2D((ty+2),tx+1)] += sx00 * confx[coo2D( fy2,fx1)] + sy00 * confy[coo2D( fy2,fx1)] + sz00 * confz[coo2D( fy2,fx1)] +
                              sx01 * confx[coo2D( fy2,fx2)] + sy01 * confy[coo2D( fy2,fx2)] + sz01 * confz[coo2D( fy2,fx2)] +
                              sx02 * confx[coo2D( fy2,fx3)] + sy02 * confy[coo2D( fy2,fx3)] + sz02 * confz[coo2D( fy2,fx3)] +
                              sx10 * confx[coo2D( fy3,fx1)] + sy10 * confy[coo2D( fy3,fx1)] + sz10 * confz[coo2D( fy3,fx1)] +
                              sx11 * confx[coo2D( fy3,fx2)] + sy11 * confy[coo2D( fy3,fx2)] + sz11 * confz[coo2D( fy3,fx2)] +
                              sx12 * confx[coo2D( fy3,fx3)] + sy12 * confy[coo2D( fy3,fx3)] + sz12 * confz[coo2D( fy3,fx3)] +
                              sx20 * confx[coo2D( fy4,fx1)] + sy20 * confy[coo2D( fy4,fx1)] + sz20 * confz[coo2D( fy4,fx1)] +
                              sx21 * confx[coo2D( fy4,fx2)] + sy21 * confy[coo2D( fy4,fx2)] + sz21 * confz[coo2D( fy4,fx2)] +
                              sx22 * confx[coo2D( fy4,fx3)] + sy22 * confy[coo2D( fy4,fx3)] + sz22 * confz[coo2D( fy4,fx3)];
  corr[coo2D((ty+2),tx+2)] += sx00 * confx[coo2D( fy2,fx2)] + sy00 * confy[coo2D( fy2,fx2)] + sz00 * confz[coo2D( fy2,fx2)] +
                              sx01 * confx[coo2D( fy2,fx3)] + sy01 * confy[coo2D( fy2,fx3)] + sz01 * confz[coo2D( fy2,fx3)] +
                              sx02 * confx[coo2D( fy2,fx4)] + sy02 * confy[coo2D( fy2,fx4)] + sz02 * confz[coo2D( fy2,fx4)] +
                              sx10 * confx[coo2D( fy3,fx2)] + sy10 * confy[coo2D( fy3,fx2)] + sz10 * confz[coo2D( fy3,fx2)] +
                              sx11 * confx[coo2D( fy3,fx3)] + sy11 * confy[coo2D( fy3,fx3)] + sz11 * confz[coo2D( fy3,fx3)] +
                              sx12 * confx[coo2D( fy3,fx4)] + sy12 * confy[coo2D( fy3,fx4)] + sz12 * confz[coo2D( fy3,fx4)] +
                              sx20 * confx[coo2D( fy4,fx2)] + sy20 * confy[coo2D( fy4,fx2)] + sz20 * confz[coo2D( fy4,fx2)] +
                              sx21 * confx[coo2D( fy4,fx3)] + sy21 * confy[coo2D( fy4,fx3)] + sz21 * confz[coo2D( fy4,fx3)] +
                              sx22 * confx[coo2D( fy4,fx4)] + sy22 * confy[coo2D( fy4,fx4)] + sz22 * confz[coo2D( fy4,fx4)];
  __syncthreads();
}

__global__ void sumcorrTRI(double *DSum_corr, const float *corr, int *DTo){
  //Energy variables
  const int x = threadIdx.x % (BlockSize_x);
  const int y = (threadIdx.x / BlockSize_x);
  const int tx = 3 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
  const int ty =(blockIdx.x / BN) * SpinSize +  3 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
  const int ty_pt =(DTo[blockIdx.x / BN]) * SpinSize +  3 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
  //calculate all the final position first
  DSum_corr[coo2D(ty_pt,tx)] += corr[coo2D(ty,tx)]/SpinSize/SpinSize;
  DSum_corr[coo2D(ty_pt,tx+1)] += corr[coo2D(ty,tx+1)]/SpinSize/SpinSize;
  DSum_corr[coo2D(ty_pt,tx+2)] += corr[coo2D(ty,tx+2)]/SpinSize/SpinSize;
  DSum_corr[coo2D((ty_pt + 1),tx)] += corr[coo2D((ty + 1),tx)]/SpinSize/SpinSize;
  DSum_corr[coo2D((ty_pt + 1),tx+1)] += corr[coo2D((ty + 1),tx+1)]/SpinSize/SpinSize;
  DSum_corr[coo2D((ty_pt + 1),tx+2)] += corr[coo2D((ty + 1),tx+2)]/SpinSize/SpinSize;
  DSum_corr[coo2D((ty_pt + 2),tx)] += corr[coo2D((ty + 2),tx)]/SpinSize/SpinSize;
  DSum_corr[coo2D((ty_pt + 2),tx+1)] += corr[coo2D((ty + 2),tx+1)]/SpinSize/SpinSize;
  DSum_corr[coo2D((ty_pt + 2),tx+2)] += corr[coo2D((ty + 2),tx+2)]/SpinSize/SpinSize;
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
  const int x = threadIdx.x % (BlockSize_x);
  const int y = (threadIdx.x / BlockSize_x);
  const int tx = 3 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
  const int ty =(blockIdx.x / BN) * SpinSize +  3 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
  //calculate all the final position first
  DSum_corr[coo2D(ty,tx)] = DSum_corr[coo2D(ty,tx)]/N_corr;
  DSum_corr[coo2D(ty,tx+1)] = DSum_corr[coo2D(ty,tx+1)]/N_corr;
  DSum_corr[coo2D(ty,tx+2)] = DSum_corr[coo2D(ty,tx+2)]/N_corr;
  DSum_corr[coo2D((ty + 1),tx)] = DSum_corr[coo2D((ty + 1),tx)]/N_corr;
  DSum_corr[coo2D((ty + 1),tx+1)] = DSum_corr[coo2D((ty + 1),tx+1)]/N_corr;
  DSum_corr[coo2D((ty + 1),tx+2)] = DSum_corr[coo2D((ty + 1),tx+2)]/N_corr;
  DSum_corr[coo2D((ty + 2),tx)] = DSum_corr[coo2D((ty + 2),tx)]/N_corr;
  DSum_corr[coo2D((ty + 2),tx+1)] = DSum_corr[coo2D((ty + 2),tx+1)]/N_corr;
  DSum_corr[coo2D((ty + 2),tx+2)] = DSum_corr[coo2D((ty + 2),tx+2)]/N_corr;
  __syncthreads();
}
#endif
