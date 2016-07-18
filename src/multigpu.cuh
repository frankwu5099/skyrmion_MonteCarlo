#ifndef MULTIGPU_H
#define MULTIGPU_H
template<class T>
class mgpudata<T>{
  T** data;
  mgpudata(size_t size, int *fdevice, int num_fdevice);
};
mgpudata::mgpudata(size_t &size, int *fdevice, int &num_fdevice){
  data = (T**)malloc(sizeof(T*)*num_fdevice);
  for (int i =0; i<num_fdevice; i++){
    cudaSetDevice(fdevice[i]);
    CudaSafeCall(cudaMalloc((void**)&data[i], size/num_fdevice);
  }
}
template<class T>
void mgpuHostToDevice(mgpudata<T> Ddata, T* Hdata, size_t &size, int* fdevice, int &num_fdevice){
  size_t single_size = size/num_fdevice;
  size_t single_num = single_size/sizeof(T);
  for (int i =0; i<num_fdevice; i++){
    cudaSetDevice(fdevice[i]);
    CudaSafeCall(cudaMemcpyAsync(Ddata.data[i], Hdata+i*single_num, single_size,cudaMemcpyHostToDevice);
  }
}
template<class T>
void mgpuDeviceToHost(mgpudata<T> Ddata, T* Hdata, size_t &size, int* fdevice, int &num_fdevice){
  size_t single_size = size/num_fdevice;
  size_t single_num = single_size/sizeof(T);
  for (int i =0; i<num_fdevice; i++){
    cudaSetDevice(fdevice[i]);
    CudaSafeCall(cudaMemcpyAsync(Hdata+i*single_num, Ddata.data[i], single_size,cudaMemcpyDeviceToHost);
  }
}
#endif
