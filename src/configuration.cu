#ifndef CONFIGURATION_H
#define CONFIGURATION_H
#include "configuration.cuh"
#endif


configuration.configuration(int Pnum, char* conf_dir){
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
  if(cudaMalloc((void**)&Dx, Spin_mem_size)){
    fprintf(stderr, "Error couldn't allocate Device Memory (x)!!\n");
    exit(1);
  }
  if(cudaMalloc((void**)&Dy, Spin_mem_size)){
    fprintf(stderr, "Error couldn't allocate Device Memory (y)!!\n");
    exit(1);
  }
  if(cudaMalloc((void**)&Dz, Spin_mem_size)){
    fprintf(stderr, "Error couldn't allocate Device Memory (z)!!\n");
    exit(1);
  }
}
void initialize (bool order){
	if (order == 0){
		double pi = 3.141592653589793;
		double th, phi;
		for(int i = 0; i < spins_num; i++){
			th = uni01_sampler() * pi;
			phi = uni01_sampler() * 2 * pi;
			Hconfx[i] = cos(th);
			th = sin(th);
			Hconfy[i] = th * cos(phi);
			Hconfz[i] = th * sin(phi);
		}
	}
	else {
		for(int i = 0; i < spins_num; i++){
			Hconfx[i] = 0;
			Hconfy[i] = 0;
			Hconfz[i] = 1;
		}
	}
  cudaMemcpy(Dx, Hx, Spin_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(Dy, Hy, Spin_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(Dz, Hz, Spin_mem_size, cudaMemcpyHostToDevice);
}
void configuration.backtoHost(){
	cudaMemcpy(Hx, Dx, Spin_mem_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(Hy, Dy, Spin_mem_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(Hz, Dz, Spin_mem_size, cudaMemcpyDeviceToHost);
  free(Hx);
  free(Hy);
  free(Hz);
  cudaFree(Dx);
  cudaFree(Dy);
  cudaFree(Dz);
  //cudaFree(Dcorr);
}
void configuration.write(){
	write(Confxfd, Hconfx, Spin_mem_size);
	write(Confyfd, Hconfy, Spin_mem_size);
	write(Confzfd, Hconfz, Spin_mem_size);
}

configuration.~configuration(){
  close(Confxfd);
  close(Confyfd);
  close(Confzfd);
}
