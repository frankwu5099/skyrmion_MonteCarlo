#include <math.h>
#include <cuda.h>
#include "WarpStandard.cuh"
#define 2D
//#define TRI
#define SQ
#define NORM (float(4.656612873077393e-10)) // UINT_MAX * NORM = 2
#define TOPI (float(1.462918078360668e-9))
#define TWOPI (float(6.28318530717956))	//2*pi
#define Hfinal (0.016000)
#define A (0.171573)//(-DR * DR)
#define DR (0.0)//(1.41421)//(0.585786)//(0.8)////(0.4)//(1.02749)
#define DD (0.585786)//(1.41421)//(0.8)////(0.4)//(1.02749)
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

#define BXPzx (-DR)
#define BYPzx (DD)
#define BXMzx (DR)
#define BYMzx (-DD)
#define BXPxz (DR)
#define BYPxz (-DD)
#define BXMxz (-DR)
#define BYMxz (DD)
//--------variables for one temperature replica----------
#define SpinSize 32                          //Each thread controls 2 by 2 by 2 spins
#define SpinSize_z 8
#define BlockSize_x 16
#define BlockSize_y 16
#define N (SpinSize*SpinSize*(SpinSize_z))              //The number of spins of the system + boundary effective spins
#define Nplane (SpinSize*SpinSize)              //The number of spins of the system
#ifdef TRI
#define GridSize_x (SpinSize/BlockSize_x/3)
#define GridSize_y (SpinSize/BlockSize_y/3)
#define TN (Nplane / 9)									//The number of needed threads
#endif
#ifdef SQ
#define GridSize_x (SpinSize/BlockSize_x/2)
#define GridSize_y (SpinSize/BlockSize_y/2)
#define TN (Nplane / 4)									//The number of needed threads
#endif
#define BN (GridSize_x*GridSize_y)       //The number of needed blocks
//---------------------End-------------------------------
