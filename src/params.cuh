#ifndef PARAMS_H
#define PARAMS_H
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <fcntl.h>
#include <stdint.h>
#include <cuda.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "helper_timer.h"
//#include "WarpStandard.cuh"
#include <time.h>
#include <string>
#include <fstream>
#include <functional>
#include <random>
#include <nlohmann/json.hpp>


//==================== cuda error checker ====================
#define CUDA_ERROR_CHECK
#define E_lowest float(-4.8)
#define E_highest float(-3.5)
#define Chern_lowest float(0.0)//(-10.5*4.0*3.1415926536)
#define Chern_highest float(1.0) //(89.5*4.0*3.1415926536)
#define Slice_NUM int(1200)
#define Slice_CNUM int(1200)

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
	file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
#endif
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  /*err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
	file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }*/
#endif
  return;
}
//=========================================================================
using namespace std;
using json = nlohmann::json;


//#deinfe THIN
#define TRI
//#define ZPERIODIC
//#define SQ
#define ORDER 0

//control the changing parameter and the parallel parameter
#define exchangecriterion(i) ((Ms[i + 1] - Ms[i]) * ( Hls[i] - Hls[i + 1]) * invT)  //for parallel on H


#define NORM (float(4.656612873077393e-10)) // UINT_MAX * NORM = 2
#define TOPI (float(1.462918078360668e-9))
#define TWOPI (float(6.28318530717956))	//2*pi
#define Hfinal (0.016000)

#ifdef SQ
#define BXPxx (1.000000)
#define BYPxx (1.000000)
#define BXMxx (1.000000)
#define BYMxx (1.000000)
#define BXPyy (1.000000)
#define BYPyy (1.000000)
#define BXMyy (1.000000)
#define BYMyy (1.000000)
#define BXPzz (1.000000)
#define BYPzz (1.000000)
#define BXMzz (1.000000)
#define BYMzz (1.000000)

#define BXPxy (0.000000)
#define BYPxy (0.000000)
#define BXMxy (0.000000)
#define BYMxy (0.000000)
#define BXPyx (0.000000)
#define BYPyx (0.000000)
#define BXMyx (0.000000)
#define BYMyx (0.000000)
#define BZPxx (1.000000)
#define BZPyy (1.000000)
#define BZPzz (1.000000)
#define BZMxx (1.000000)
#define BZMyy (1.000000)
#define BZMzz (1.000000)


#define BXPyz (DD)
#define BYPyz (DR)
#define BXMyz (-DD)
#define BYMyz (-DR)
#define BXPzy (-DD)
#define BYPzy (-DR)
#define BXMzy (DD)
#define BYMzy (DR)

#define BXPxz (DR)
#define BYPxz (-DD)
#define BXMxz (-DR)
#define BYMxz (DD)
#define BXPzx (-DR)
#define BYPzx (DD)
#define BXMzx (DR)
#define BYMzx (-DD)
#endif

#ifdef THIN
#define BXPxx (1.000000)
#define BYPxx (1.000000)
#define BZPxx (1.000000)
#define BXMxx (1.000000)
#define BYMxx (1.000000)
#define BZMxx (1.000000)
#define BXPyy (1.000000)
#define BYPyy (1.000000)
#define BZPyy (1.000000)
#define BXMyy (1.000000)
#define BYMyy (1.000000)
#define BZMyy (1.000000)
#define BXPzz (1.000000)
#define BYPzz (1.000000)
#define BZPzz (1.000000)
#define BXMzz (1.000000)
#define BYMzz (1.000000)
#define BZMzz (1.000000)

#define BXPxy (0.000000)
#define BYPxy (0.000000)
#define BXMxy (0.000000)
#define BYMxy (0.000000)
#define BXPyx (0.000000)
#define BYPyx (0.000000)
#define BXMyx (0.000000)
#define BYMyx (0.000000)

#define BZPyx (-DD)
#define BZMyx (DD)
#define BZPxy (DD)
#define BZMxy (-DD)

#define BXPyz (DD)
#define BYPyz (DR)
#define BXMyz (-DD)
#define BYMyz (-DR)
#define BXPzy (-DD)
#define BYPzy (-DR)
#define BXMzy (DD)
#define BYMzy (DR)

#define BXPxz (DR)
#define BYPxz (-DD)
#define BXMxz (-DR)
#define BYMxz (DD)
#define BXPzx (-DR)
#define BYPzx (DD)
#define BXMzx (DR)
#define BYMzx (-DD)
#endif

#ifdef TRI
#define sqrt3d2 (0.866025f)
#define BXPxx (1.000000)
#define BYPxx (1.000000)
#define BWPxx (1.000000)
#define BZPxx (1.000000)
#define BXMxx (1.000000)
#define BYMxx (1.000000)
#define BWMxx (1.000000)
#define BZMxx (1.000000)
#define BXPyy (1.000000)
#define BYPyy (1.000000)
#define BWPyy (1.000000)
#define BZPyy (1.000000)
#define BXMyy (1.000000)
#define BYMyy (1.000000)
#define BWMyy (1.000000)
#define BZMyy (1.000000)
#define BXPzz (1.000000)
#define BYPzz (1.000000)
#define BWPzz (1.000000)
#define BZPzz (1.000000)
#define BXMzz (1.000000)
#define BYMzz (1.000000)
#define BWMzz (1.000000)
#define BZMzz (1.000000)

#define BXPxy (0.000000)
#define BYPxy (0.000000)
#define BWPxy (0.000000)
#define BXMxy (0.000000)
#define BYMxy (0.000000)
#define BWMxy (0.000000)
#define BXPyx (0.000000)
#define BYPyx (0.000000)
#define BWPyx (0.000000)
#define BXMyx (0.000000)
#define BYMyx (0.000000)
#define BWMyx (0.000000)

#endif

#define MEASURE_NUM 12



//---------------------coo-----------------------------
#define cals_coo(k, j, i) ((k) * cals_Nplane + (j) * cals_SpinSize + (i))
#define cals_coo2D(j, i) ((j) * cals_SpinSize + (i))
#define corr_coo(k, j, i) ((k) * corr_Nplane + (j) * corr_SpinSize + (i))
#define corr_coo2D(j, i) ((j) * corr_SpinSize + (i))
#define flip_coo(k, j, i) ((k) * flip_Nplane + (j) * flip_SpinSize + (i))
#define flip_coo2D(j, i) ((j) * flip_SpinSize + (i))
//---------------------coo-----------------------------


//------------ alias of kernel functions --------------

#ifdef THIN
#define CAL(confx, confy, confz, out) calthin<<<grid, block, caloutputsize>>>(confx, confy, confz, out);
#define GETCORR(confx, confy, confz, corr, i, j) getcorrthin<<<grid, block>>>(confx, confy, confz, corr, i, j);
#endif
#ifdef TRI
#define CAL(confx, confy, confz, out,stream_i) calTRI<<<grid, block, caloutputsize,stream_i>>>(confx, confy, confz, out);
#define GETCORR(confx, confy, confz, corr, i, j,stream_i) getcorrTRI<<<grid, block,0,stream_i>>>(confx, confy, confz, corr, i, j);
#define GETCORR_SK(skyr_den, corr, i, j ,stream_i) getcorrTRI<<<grid, block,0,stream_i>>>(skyr_den, corr, i, j);
#define GETSKYRDEN(confx, confy, confz, skyr_den, stream_i) skyr_den_gen<<<grid, block, 0, stream_i>>>(confx, confy, confz, skyr_den);
#define SSF1(confx, confy, confz, hs, invTs, stream_i) {  flip1_TRI<<<grid, block,0 , stream_i>>>(confx, confy, confz, hs, invTs, devStates);CudaCheckError();}
#define SSF2(confx, confy, confz, hs, invTs, stream_i) {  flip2_TRI<<<grid, block,0 , stream_i>>>(confx, confy, confz, hs, invTs, devStates);CudaCheckError();}
#define SSF3(confx, confy, confz, hs, invTs, stream_i) {  flip3_TRI<<<grid, block,0 , stream_i>>>(confx, confy, confz, hs, invTs, devStates);CudaCheckError();}
//#define SSF(confx, confy, confz, rng, hs, invT) {  flip1_TRI<<<grid, block>>>(confx, confy, confz, rng, hs, invT);CudaCheckError();\
//  flip2_TRI<<<grid, block>>>(confx, confy, confz, rng, hs, invT);CudaCheckError();\
//  flip3_TRI<<<grid, block>>>(confx, confy, confz, rng, hs, invT);CudaCheckError();}
#endif
#ifdef SQ
#define CAL(confx, confy, confz, out) cal2D<<<grid, block, caloutputsize>>>(confx, confy, confz, out);
#define GETCORR(confx, confy, confz, corr, i, j) getcorr2D<<<grid, block>>>(confx, confy, confz, corr, i, j);
#endif

extern unsigned int grid;
extern unsigned int block;
extern unsigned int rngShmemsize;
extern unsigned int caloutputsize;

extern unsigned int H_SpinSize;
extern unsigned int H_SpinSize_z;
extern unsigned int H_BlockSize_x;
extern unsigned int H_BlockSize_y;
extern unsigned int H_GridSize_x;
extern unsigned int H_GridSize_y;
extern unsigned int H_N;
extern unsigned int H_Nplane;
extern unsigned int H_TN;
extern unsigned int H_BN;
//------ system size setting end --------
//------ gpu setting -------
extern int StreamN;
extern int device_0;
extern cudaStream_t stream[10];
//---- gpu setting end -----
//------ system variable setting --------
//!!!!!!!!!!!!notice that the value of DD and DR are set while compile for the efficiency of triangular lattic.
extern float H_A; //(0.0)
extern float DD; //(0.0)
extern float DR; //(0.0)
extern float H_Q1x; //(0.0)
extern float H_Q1y; //(0.0)
extern float H_Q2x; //(0.0)
extern float H_Q2y; //(0.0)
//----- system variable setting end ------

//----- simulation setting ------
extern unsigned int BIN_SZ;
extern unsigned int BIN_NUM;
extern unsigned int EQUI_N;
extern unsigned int EQUI_Ni;//(4000)//0)
extern unsigned int relax_N;
extern float PTF;             //Frequency of parallel tempering
extern unsigned int f_CORR;
extern unsigned int CORR_N;
extern unsigned int Pnum;
extern unsigned int Cnum;
extern unsigned int Tnum;
extern unsigned int Hnum;
extern vector< vector<float> > Tls;
extern vector< vector<float> > Hls;
extern string Output;  //set the output directory
//#define BIN_SZ 3000//0//00//
//#define BIN_NUM 3//0
//#define EQUI_N 20000//0//0//00////16000000
void read_params(char* param_file);
json read_json(char* param_file);

//----- simulation setting -----

#endif
