#ifndef Warp
#define Warp
#include <stdint.h>

/////////////////////////////////////////////////////////////////////////////////////
// Public constants

#define WarpStandard_K 32
#define WarpStandard_REG_COUNT 4
#define WarpStandard_STATE_WORDS 32
//#define WarpStandard_name "WarpRNG[CorrelatedU32Rng;k=32;g=16;rs=0;w=32;n=1024;hash=deac2e12ec6e615]"


//////////////////////////////////////////////////////////////////////////////////////
// Private constants


////////////////////////////////////////////////////////////////////////////////////////
// Public functions

__device__ void WarpStandard_LoadState(const unsigned *seed, unsigned *regs, unsigned *shmem);

__device__ void WarpStandard_SaveState(const unsigned *regs, const unsigned *shmem, unsigned *seed);
__device__ unsigned WarpStandard_Generate(unsigned *regs, unsigned *shmem);
#endif
