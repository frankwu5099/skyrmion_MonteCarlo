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
#include <helper_timer.h>
#include "WarpStandard.cuh"
#include <time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>


//==================== cuda error checker ====================
#define CUDA_ERROR_CHECK

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
  err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
	file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
#endif
  return;
}
//=========================================================================
using namespace boost;
extern unsigned seed;
extern mt19937 rng;
extern uniform_01<mt19937> uni01_sampler;
//#deinfe THIN
#define TRI
//#define SQ
#define ORDER 0

//control the changing parameter and the parallel parameter
#define Cparameters invTs   //control everything
#define Cparameter invT   //control everything
#define Pparameters Hs
#define Pparameter H
#define HPparameters HHs
#define DPparameters DHs
#define Cnum Tnum
#define Pnum Hnum
#define C_mem_size Temp_mem_size
#define P_mem_size H_mem_size
#define Porder(i)  Hls[Po[i]] //1.0/Tls[Po[i]]
#define Ccurrent(i) (1/Tls[Cnum - 1 - i])
#define exchangecriterion(i) ((Ms[i + 1] - Ms[i]) * ( Hls[i] - Hls[i + 1]) * invT)  //for parallel on H


#define NORM (float(4.656612873077393e-10)) // UINT_MAX * NORM = 2
#define TOPI (float(1.462918078360668e-9))
#define TWOPI (float(6.28318530717956))	//2*pi
#define Hfinal (0.016000)
#define A (0.171573)//(-DR * DR)
#define DR (0.0)//(1.41421)//(0.585786)//(0.8)////(0.4)//(1.02749)
#define DD (0.585786)//(1.41421)//(0.8)////(0.4)//(1.02749)

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

#define BXPyz (DD)
#define BYPyz (-0.5 * DD - sqrt(3.0)/2.0 * DR)
#define BWPyz (0.5 * DD - sqrt(3.0)/2.0 * DR)
#define BXMyz (-DD)
#define BYMyz (0.5 * DD + sqrt(3.0)/2.0 * DR)
#define BWMyz (-0.5 * DD + sqrt(3.0)/2.0 * DR)
#define BXPzy (-DD)
#define BYPzy  (0.5 * DD + sqrt(3.0)/2.0 * DR)
#define BWPzy  (-0.5 * DD + sqrt(3.0)/2.0 * DR) 
#define BXMzy (DD)
#define BYMzy (-0.5 * DD - sqrt(3.0)/2.0 * DR) 
#define BWMzy (0.5 * DD - sqrt(3.0)/2.0 * DR)

#define BXPxz (-DR)
#define BYPxz (-sqrt(3.0)/2.0 * DD + 0.5 * DR)
#define BWPxz (-sqrt(3.0)/2.0 * DD - 0.5 * DR)
#define BXMxz (DR)
#define BYMxz (sqrt(3.0)/2.0 * DD - 0.5 * DR)
#define BWMxz (sqrt(3.0)/2.0 * DD + 0.5 * DR)
#define BXPzx (DR)
#define BYPzx (sqrt(3.0)/2.0 * DD - 0.5 * DR)
#define BWPzx (sqrt(3.0)/2.0 * DD + 0.5 * DR)
#define BXMzx (-DR)
#define BYMzx (-sqrt(3.0)/2.0 * DD + 0.5 * DR)
#define BWMzx (-sqrt(3.0)/2.0 * DD - 0.5 * DR)
#endif
//--------variables for one temperature replica----------
#define SpinSize 48                          //Each thread controls 2 by 2 by 2 spins
#define SpinSize_z 1
#define BlockSize_x 16
#define BlockSize_y 16
#define N (SpinSize*SpinSize*(SpinSize_z))      //The number of spins of the system
#define Nplane (SpinSize*SpinSize)              //The number of spins of the system

#ifdef THIN
#define GridSize_x (SpinSize/BlockSize_x/2)
#define GridSize_y (SpinSize/BlockSize_y/2)
#define TN (Nplane / 4)				//The number of needed threads
#endif

#ifdef SQ
#define GridSize_x (SpinSize/BlockSize_x/2)
#define GridSize_y (SpinSize/BlockSize_y/2)
#define TN (Nplane / 4)				//The number of needed threads
#endif

#ifdef TRI
#define GridSize_x (SpinSize/BlockSize_x/3)
#define GridSize_y (SpinSize/BlockSize_y/3)
#define TN (Nplane / 9)				//The number of needed threads
#endif

#define BN (GridSize_x*GridSize_y)       //The number of needed blocks per replica
#define MEASURE_NUM 5
//---------------------End-------------------------------



//---------------------coo-----------------------------
#define coo(k, j, i) ((k) * Nplane + (j) * SpinSize + (i))
#define coo2D(j, i) ((j) * SpinSize + (i))
//---------------------coo-----------------------------


//------------ alias of kernel functions --------------

#ifdef THIN
#define CAL(confx, confy, confz, out) calthin<<<grid, block>>>(confx, confy, confz, out);
#define GETCORR(confx, confy, confz, corr, i, j) getcorrthin<<<grid, block>>>(confx, confy, confz, corr, i, j);
#endif
#ifdef TRI
#define CAL(confx, confy, confz, out) calTRI<<<grid, block>>>(confx, confy, confz, out);
#define GETCORR(confx, confy, confz, corr, i, j) getcorrTRI<<<grid, block>>>(confx, confy, confz, corr, i, j);
#endif
#ifdef SQ
#define CAL(confx, confy, confz, out) cal2D<<<grid, block>>>(confx, confy, confz, out);
#define GETCORR(confx, confy, confz, corr, i, j) getcorr2D<<<grid, block>>>(confx, confy, confz, corr, i, j);
#endif

extern unsigned int grid;
extern unsigned int block;
#endif
