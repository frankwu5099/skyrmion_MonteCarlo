#ifndef PARAMS_H
#define PARAMS_H
#include "params.cuh"
#endif
int setDev(){
  int num_devices,flag=0;
  cudaError_t error = cudaErrorDevicesUnavailable;
  cudaGetDeviceCount(&num_devices);
  bool *Dtest;
  for (int device = 0; device < num_devices; device++) {//
    cudaSetDevice(device);
    error = cudaMalloc((void**)&Dtest, 100*sizeof(bool));
    if (error == cudaSuccess){
      printf("using device %d !\n",device);
      cudaFree(Dtest);
      break;

    }else{
      /* if GPU busy*/
      if (error == cudaErrorDevicesUnavailable){
	printf("device %d is busy >> try another!\n",device);
      }else{
	printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
	printf("CUDA error: %s\n", cudaGetErrorString(error));
      }
      if(device==num_devices-1)
	{printf("%s\n","ERROR! no avalible device now!");flag=1;}
    }

  }
  return flag ;
}


/*
void debugg(float *x,float *y,float *ang){

  for(int i=25;i<40;i++){

    printf("%d-Hang: %f\tHcos: %f\tHsin: %f\ngetA: %f\n",i,ang[i],x[i],y[i],getangf(x[i],y[i],sqrt(x[i]*x[i]+y[i]*y[i])));


  }


}
*/
