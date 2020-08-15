#include "configuration.cuh"


configuration::configuration(int Pnum, char* conf_dir){
	f_index = 0;
  configurations_num = Pnum;
  configurations_num_s = Pnum/StreamN;
  Spin_mem_size = configurations_num * H_N * sizeof(float);
  Single_mem_size = H_N * sizeof(float);
  spins_num = configurations_num * H_N;
  Spin_mem_size_s = configurations_num_s * H_N * sizeof(float);
  spins_num_s = configurations_num_s * H_N;
  sprintf(dirfn, "%s", conf_dir);
  sprintf(Confxfn, "%s/Confx_%d", dirfn, f_index);
  sprintf(Confyfn, "%s/Confy_%d", dirfn, f_index);
  sprintf(Confzfn, "%s/Confz_%d", dirfn, f_index);
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
    /*
    for(int i = 0; i < spins_num; i++){
      Hx[i] = 0;
      Hy[i] = 0;
      Hz[i] = 1;
    }*/
    double pi = 3.141592653589793;
    double th, phi;
    for(int i_P = 0; i_P < configurations_num; i_P++){
      for(int iz = 0; iz < H_SpinSize_z; iz++){
        for(int ix = 0; ix < H_SpinSize; ix++){
          for(int iy = 0; iy < H_SpinSize; iy++){
            phi = (ix-0.5*iy) * pi/3.0;
            Hx[H_N * i_P + H_Nplane * iz + H_SpinSize * iy + ix] = 0;
            Hy[H_N * i_P + H_Nplane * iz + H_SpinSize * iy + ix] = cos(phi);
            Hz[H_N * i_P + H_Nplane * iz + H_SpinSize * iy + ix] = sin(phi);
          }
        }
      }
    }
  }
  for (int gpu_i = 0 ; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyAsync(Dx[gpu_i], Hx + gpu_i * spins_num_s, Spin_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyAsync(Dy[gpu_i], Hy + gpu_i * spins_num_s, Spin_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyAsync(Dz[gpu_i], Hz + gpu_i * spins_num_s, Spin_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
  }
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    cudaDeviceSynchronize();
  }
}
void configuration::backtoHost(){
  for (int gpu_i = 0 ; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyAsync(Hx + gpu_i * spins_num_s, Dx[gpu_i], Spin_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyAsync(Hy + gpu_i * spins_num_s, Dy[gpu_i], Spin_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyAsync(Hz + gpu_i * spins_num_s, Dz[gpu_i], Spin_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));
  }
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
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
  if (f_index > 30){
    for (int i = 0; i < configurations_num; i+=2){
            write(Confzfd, Hz+(i*H_N), H_N * sizeof(float));
            write(Confyfd, Hy+(i*H_N), H_N * sizeof(float));
            write(Confxfd, Hx+(i*H_N), H_N * sizeof(float));
    }
  }
  close(Confxfd);
  close(Confyfd);
  close(Confzfd);
  f_index += 1;
  sprintf(Confxfn, "%s/Confx_%d", dirfn, f_index);
  sprintf(Confyfn, "%s/Confy_%d", dirfn, f_index);
  sprintf(Confzfn, "%s/Confz_%d", dirfn, f_index);
  Confxfd = open(Confxfn, O_CREAT | O_WRONLY, 0644);
  Confyfd = open(Confyfn, O_CREAT | O_WRONLY, 0644);
  Confzfd = open(Confzfn, O_CREAT | O_WRONLY, 0644);
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
