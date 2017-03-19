#include "configuration.cuh"


configuration::configuration(int Pnum, char* conf_dir){
  configurations_num = Pnum;
  configurations_num_s = Pnum/StreamN;
  Spin_mem_size = configurations_num * H_N * sizeof(float);
  Single_mem_size = H_N * sizeof(float);
  spins_num = configurations_num * H_N;
  Spin_mem_size_s = configurations_num_s * H_N * sizeof(float);
  spins_num_s = configurations_num_s * H_N;
  sprintf(Confxfn, "%s/Confx", conf_dir);
  sprintf(Confyfn, "%s/Confy", conf_dir);
  sprintf(Confzfn, "%s/Confz", conf_dir);
  Hx = (float*)malloc(Spin_mem_size);
  Hy = (float*)malloc(Spin_mem_size);
  Hz = (float*)malloc(Spin_mem_size);
  Confxfd = open(Confxfn, O_CREAT | O_WRONLY, 0644);
  Confyfd = open(Confyfn, O_CREAT | O_WRONLY, 0644);
  Confzfd = open(Confzfn, O_CREAT | O_WRONLY, 0644);
  Dx = (float**)calloc(StreamN, sizeof(float*));
  Dy = (float**)calloc(StreamN, sizeof(float*));
  Dz = (float**)calloc(StreamN, sizeof(float*));
  for (int gpu_i = 0 ; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMalloc((void**)&Dx[gpu_i], Spin_mem_size_s));
    CudaSafeCall(cudaMalloc((void**)&Dy[gpu_i], Spin_mem_size_s));
    CudaSafeCall(cudaMalloc((void**)&Dz[gpu_i], Spin_mem_size_s));
  }
}



void configuration::initialize (bool order){
  if (order == 0){
    double pi = 3.141592653589793;
    double th, phi;
    for(int i = 0; i < spins_num; i++){
      th = uni01_sampler() * pi;
      phi = uni01_sampler() * 2 * pi;
      Hx[i] = cos(th);
      th = sin(th);
      Hy[i] = th * cos(phi);
      Hz[i] = th * sin(phi);
    }
  }
  else {
    for(int i = 0; i < spins_num; i++){
      Hx[i] = 0;
      Hy[i] = 0;
      Hz[i] = 1;
    }
  }
  for (int gpu_i = 0 ; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyasync(Dx[gpu_i], Hx + gpu_i * spins_nums, Spin_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyasync(Dy[gpu_i], Hy + gpu_i * spins_nums, Spin_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyasync(Dz[gpu_i], Hz + gpu_i * spins_nums, Spin_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
  }
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    cudaDeviceSynchronize();
  }
}
void configuration::backtoHost(){
  for (int gpu_i = 0 ; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyasync(Hx + gpu_i * spins_nums, Dx[gpu_i], Spin_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyasync(Hy + gpu_i * spins_nums, Dy[gpu_i], Spin_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyasync(Hz + gpu_i * spins_nums, Dz[gpu_i], Spin_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));
  }
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    cudaDeviceSynchronize();
  }
}

void configuration::Dominatestateback(int hostid, int deviceid){
  cudaSetDevice(device_0 + deviceid/configurations_num_s);
  CudaSafeCall(cudaMemcpy(((float*)Hx) + hostid * H_N, ((float*)Dx[deviceid/configurations_num_s]) + (deviceid%configurations_num_s) * H_N, Single_mem_size, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(((float*)Hy) + hostid * H_N, ((float*)Dy[deviceid/configurations_num_s]) + (deviceid%configurations_num_s) * H_N, Single_mem_size, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(((float*)Hz) + hostid * H_N, ((float*)Dz[deviceid/configurations_num_s]) + (deviceid%configurations_num_s) * H_N, Single_mem_size, cudaMemcpyDeviceToHost));
  //cudaFree(Dcorr);
}
void configuration::writedata(){
  write(Confxfd, Hx, Spin_mem_size);
  write(Confyfd, Hy, Spin_mem_size);
  write(Confzfd, Hz, Spin_mem_size);
}

configuration::~configuration(){
  printf("conf free begin!\n");
  fflush(stdout);
  free(Hx);
  free(Hy);
  free(Hz);
  for (int gpu_i = 0 ; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaFree(Dx[gpu_i]));
    CudaSafeCall(cudaFree(Dy[gpu_i]));
    CudaSafeCall(cudaFree(Dz[gpu_i]));
  }
  close(Confxfd);
  close(Confyfd);
  close(Confzfd);
  printf("conf free succeed!\n");
  fflush(stdout);
}
