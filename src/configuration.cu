#include "configuration.cuh"


configuration::configuration(int Pnum, char* conf_dir){
  Spin_mem_size = Pnum * N * sizeof(float);
  spins_num = Pnum * N;
  configurations_num = Pnum;
  sprintf(Confxfn, "%s/Confx", conf_dir);
  sprintf(Confyfn, "%s/Confy", conf_dir);
  sprintf(Confzfn, "%s/Confz", conf_dir);
  Hx = (float*)malloc(Spin_mem_size);
  Hy = (float*)malloc(Spin_mem_size);
  Hz = (float*)malloc(Spin_mem_size);
  Confxfd = open(Confxfn, O_CREAT | O_WRONLY, 0644);
  Confyfd = open(Confyfn, O_CREAT | O_WRONLY, 0644);
  Confzfd = open(Confzfn, O_CREAT | O_WRONLY, 0644);
  CudaSafeCall(cudaMalloc((void**)&Dx, Spin_mem_size));
  CudaSafeCall(cudaMalloc((void**)&Dy, Spin_mem_size));
  CudaSafeCall(cudaMalloc((void**)&Dz, Spin_mem_size));
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
  CudaSafeCall(cudaMemcpy(Dx, Hx, Spin_mem_size, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(Dy, Hy, Spin_mem_size, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(Dz, Hz, Spin_mem_size, cudaMemcpyHostToDevice));
}
void configuration::backtoHost(){
  CudaSafeCall(cudaMemcpy(Hx, Dx, Spin_mem_size, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(Hy, Dy, Spin_mem_size, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(Hz, Dz, Spin_mem_size, cudaMemcpyDeviceToHost));
  free(Hx);
  free(Hy);
  free(Hz);
  CudaSafeCall(cudaFree(Dx));
  CudaSafeCall(cudaFree(Dy));
  CudaSafeCall(cudaFree(Dz));
  //cudaFree(Dcorr);
}
void configuration::writedata(){
  write(Confxfd, Hx, Spin_mem_size);
  write(Confyfd, Hy, Spin_mem_size);
  write(Confzfd, Hz, Spin_mem_size);
}

configuration::~configuration(){
  close(Confxfd);
  close(Confyfd);
  close(Confzfd);
}
