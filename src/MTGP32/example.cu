/* * This program uses the device CURAND API to calculate what * proportion of pseudo-random ints have low bit set. */ 
#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h> 
#include <iostream>
#include "port_mtgp32_host.h"  ///host part include this header 

 
//////////////////////////////////////////////////////////
//Device part : 
#include "port_mtgp32_kernel.h" ///device part include this header 
__global__ void generate_kernel(curandStateMtgp32 *state) { 
	
	///Generate uint32 
	///all thread in a block should not divergent around this. 
	unsigned int r = curand(&state[blockIdx.x]);
	r = curand(&state[blockIdx.x]);
	r = curand(&state[blockIdx.x]);
	r = curand(&state[blockIdx.x]);

}
/////////////////////////////////////////////////////////


using namespace std;
int main(int argc, char * argv[]){

	//Define state:
	curandStateMtgp32 *DStates;
	mtgp32_kernel_params *DParams;
	
	//Env:
	unsigned int NBlock = 64;
	unsigned int ThreadperBlock = 256; // Max limit at 256	
	unsigned int seed = 99;
	
	//Allocate Stats:
	cudaMalloc((void**)&DStates,sizeof(curandStateMtgp32)*NBlock);
	
	//Set parameters:
	cudaMalloc((void**)&DParams, sizeof(mtgp32_kernel_params));
	

	///Make constant:
	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, DParams);
	//return 0;
        curandMakeMTGP32KernelState(DStates, mtgp32dc_params_fast_11213, DParams, NBlock, seed);


	generate_kernel<<<NBlock,ThreadperBlock>>>(DStates);

	///cleanup

	cudaFree(DStates);
	cudaFree(DParams);
	
	
	
		




}



