#include "measurements.cuh"


measurements::measurements(char * indir, int Parallel_num, unsigned int binSize){
  measurement_num = 19;
  //raw_memmory = operator new[] (measurement_num * sizeof(measurement));
  strcpy(names[0], "E");
  strcpy(names[1], "M");
  strcpy(names[2], "Chern");
  strcpy(names[3], "E2");
  strcpy(names[4], "E4");
  strcpy(names[5], "M2");
  strcpy(names[6], "M4");
  strcpy(names[7], "Mz2");
  strcpy(names[8], "Mz4");
  strcpy(names[9], "Chern2");
  strcpy(names[10], "Chern4");
  strcpy(names[11], "Nematic2");
  strcpy(names[12], "Nematic4");
  strcpy(names[13], "Mz");
  strcpy(names[14], "EMz");
  strcpy(names[15], "Nematic");
  strcpy(names[16], "Mzi2");
  strcpy(names[17], "Hy");
  strcpy(names[18], "Iy2");
  norms[0] = double(binSize) * H_N;
  norms[1] = double(binSize) * H_N;
  norms[2] = double(binSize) * 2;
  norms[3] = double(binSize) * H_N * H_N;
  norms[4] = double(binSize) * H_N * H_N * H_N * H_N;
  norms[5] = double(binSize) * H_N * H_N;
  norms[6] = double(binSize) * H_N * H_N * H_N * H_N;
  norms[7] = double(binSize) * H_N * H_N;
  norms[8] = double(binSize) * H_N * H_N * H_N * H_N;
  norms[9] = double(binSize) * H_N * H_N * 2 * 2;
  norms[10] = double(binSize) * H_N * H_N * H_N * H_N * 2 * 2 * 2 * 2;
  norms[11] = double(binSize) * H_N * H_N;
  norms[12] = double(binSize) * H_N * H_N * H_N * H_N;
  norms[13] = double(binSize) * H_N;
  norms[14] = double(binSize) * H_N * H_N;
  norms[15] = double(binSize) * H_N;
  norms[16] = double(binSize) * H_N;
  norms[17] = double(binSize) * H_N;
  norms[18] = double(binSize) * H_N * H_N;
  O.reserve(measurement_num);
  for (int i =0 ; i< measurement_num; i++){
    O.push_back(measurement(indir, names[i], norms[i], Parallel_num));
    O[O.size() - 1].fp = fopen(O[O.size() - 1].fn, "w");
  }
  data_num = Parallel_num;
  data_num_s = Parallel_num/StreamN;
  Out_mem_size = Parallel_num * MEASURE_NUM * H_BN * sizeof(double);
  Out_mem_size_s = data_num_s * MEASURE_NUM * H_BN * sizeof(double);
  printf("%u\n", Out_mem_size);
  Hout = (double*)calloc(Parallel_num * MEASURE_NUM * H_BN, sizeof(double));
  Dout = (double**)calloc(StreamN, sizeof(double*));
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMalloc((void**)&Dout[gpu_i], Out_mem_size_s));
  }
  EHistogram = (unsigned int*) calloc(Parallel_num * Slice_NUM, sizeof(unsigned int));
  ChernHistogram = (unsigned int*) calloc(Parallel_num * Slice_CNUM, sizeof(unsigned int));
  hist_start = 0;
}


measurements::~measurements(){
  printf("measure free begin!\n");
  fflush(stdout);
  for (int i =0 ; i< measurement_num; i++){
    fclose(O[i].fp);
  }
  free(Hout);
  free(EHistogram);
  free(ChernHistogram);
  //CudaSafeCall(cudaFree(Dout));
  printf("measure free succeed!\n");
  fflush(stdout);
}



void measurements::virtual_measure(float** Dconfx, float** Dconfy, float** Dconfz, std::vector<int>& Ho, double* Ms, double* Es, float* HHs){
  static int raw_off;
  static double E;
  static double Mz;
  int gpu_i;
	printf("device start : %d, #streams = %d\n" , device_0, StreamN);
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CAL(Dconfx[gpu_i], Dconfy[gpu_i], Dconfz[gpu_i], Dout[gpu_i], stream[gpu_i]);//cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout);
  }
  CudaCheckError();
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyAsync(Hout+gpu_i * data_num_s * MEASURE_NUM * H_BN, Dout[gpu_i], Out_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));//Async, stream[gpu_i]
  }
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    cudaDeviceSynchronize();
  }

  for(int t = 0; t < data_num; t++){
    raw_off = t * MEASURE_NUM * H_BN;
    E = 0;
    Mz = 0;
    for(int j = 0; j < H_BN; j++)
      E += Hout[raw_off + j];
    for(int j = 3 * H_BN; j < 4 * H_BN; j++)
      Mz += Hout[raw_off + j];
    Ms[Ho[t]] = Mz;	//Es is the energies in order of temperature set
    E = E - HHs[t] * Mz;
    Es[Ho[t]] = E;	//Es is the energies in order of temperature set
  }
}



void measurements::measure(float** Dconfx, float** Dconfy, float** Dconfz, std::vector<int>& Ho, double* Ms, double* Es, float* HHs){
  static int raw_off;
  static double E, E2;
  static double Mx, My, Mz, Chern, M2, Mz2, Chern2;
  static double Mxx, Myy, Mxy, eta, Mzi2, Hy, Iy;
  //static double spinQ1x_r, spinQ1y_r, spinQ1z_r, spinQ1x_i, spinQ1y_i, spinQ1z_i;
  //static double spinQ2x_r, spinQ2y_r, spinQ2z_r, spinQ2x_i, spinQ2y_i, spinQ2z_i;
  int gpu_i;
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CAL(Dconfx[gpu_i], Dconfy[gpu_i], Dconfz[gpu_i], Dout[gpu_i], stream[gpu_i]);//cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout);
  }
  CudaCheckError();
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyAsync(Hout + gpu_i * data_num_s * MEASURE_NUM * H_BN, Dout[gpu_i], Out_mem_size_s, cudaMemcpyDeviceToHost, stream[gpu_i]));//Async, stream[gpu_i]
  }
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    cudaDeviceSynchronize();
  }


  for(int t = 0; t < data_num; t++){
    raw_off = t * MEASURE_NUM * H_BN;
    E = 0, E2 = 0;
    Mx = 0, My = 0, Mz = 0, Chern = 0;
    Mxx = 0, Myy = 0, Mxy = 0, Iy = 0, Hy = 0, Mzi2 = 0;
    /*
    spinQ1x_r = 0, spinQ1y_r = 0, spinQ1z_r = 0;
    spinQ1x_i = 0, spinQ1y_i = 0, spinQ1z_i = 0;
    spinQ2x_r = 0, spinQ2y_r = 0, spinQ2z_r = 0;
    spinQ2x_i = 0, spinQ2y_i = 0, spinQ2z_i = 0;
    */
    for(int j = 0; j < H_BN; j++)
      E += Hout[raw_off + j];
    for(int j = H_BN; j < 2 * H_BN; j++)
      Mx += Hout[raw_off + j];
    for(int j = 2 * H_BN; j < 3 * H_BN; j++)
      My += Hout[raw_off + j];
    for(int j = 3 * H_BN; j < 4 * H_BN; j++)
      Mz += Hout[raw_off + j];
    for(int j = 4 * H_BN; j < 5 * H_BN; j++)
      Chern += Hout[raw_off + j];
    for(int j = 5 * H_BN; j < 6 * H_BN; j++)
      Mxx += Hout[raw_off + j];
    for(int j = 6 * H_BN; j < 7 * H_BN; j++)
      Myy += Hout[raw_off + j];
    for(int j = 7 * H_BN; j < 8 * H_BN; j++)
      Mxy += Hout[raw_off + j];
    for(int j = 8 * H_BN; j < 9 * H_BN; j++)
      Mzi2 += Hout[raw_off + j];
    for(int j = 9 * H_BN; j < 10 * H_BN; j++)
      Hy += Hout[raw_off + j];
    for(int j =10 * H_BN; j < 11 * H_BN; j++)
      Iy += Hout[raw_off + j];
    /*
    for(int j = 5 * H_BN; j < 6 * H_BN; j++)
      spinQ1x_r += Hout[raw_off + j];
    for(int j = 6 * H_BN; j < 7 * H_BN; j++)
      spinQ1y_r += Hout[raw_off + j];
    for(int j = 7 * H_BN; j < 8 * H_BN; j++)
      spinQ1z_r += Hout[raw_off + j];
    for(int j = 8 * H_BN; j < 9 * H_BN; j++)
      spinQ1x_i += Hout[raw_off + j];
    for(int j = 9 * H_BN; j < 10 * H_BN; j++)
      spinQ1y_i += Hout[raw_off + j];
    for(int j = 10 * H_BN; j < 11 * H_BN; j++)
      spinQ1z_i += Hout[raw_off + j];
    for(int j = 11 * H_BN; j < 12 * H_BN; j++)
      spinQ2x_r += Hout[raw_off + j];
    for(int j = 12 * H_BN; j < 13 * H_BN; j++)
      spinQ2y_r += Hout[raw_off + j];
    for(int j = 13 * H_BN; j < 14 * H_BN; j++)
      spinQ2z_r += Hout[raw_off + j];
    for(int j = 14 * H_BN; j < 15 * H_BN; j++)
      spinQ2x_i += Hout[raw_off + j];
    for(int j = 15 * H_BN; j < 16 * H_BN; j++)
      spinQ2y_i += Hout[raw_off + j];
    for(int j = 16 * H_BN; j < 17 * H_BN; j++)
      spinQ2z_i += Hout[raw_off + j];
    */
    Ms[Ho[t]] = Mz;	//Es is the energies in order of temperature set
    E = E - HHs[t] * Mz;
    Es[Ho[t]] = E;	//Es is the energies in order of temperature set
    O[0].outdata[Ho[t]] += E;
    M2 = Mx * Mx + My * My + Mz * Mz;
    Mz2 = Mz * Mz;
    Chern2 = Chern * Chern;
    E2 = E * E;
    eta = sqrt(Mxx*Mxx + Myy*Myy - 2*Mxx*Myy + 4*Mxy*Mxy);
    O[1].outdata[Ho[t]] += sqrt(M2);
    O[2].outdata[Ho[t]] += Chern;
    O[3].outdata[Ho[t]] += E2;
    O[5].outdata[Ho[t]] += M2;
    O[4].outdata[Ho[t]] += E2 * E2;
    O[6].outdata[Ho[t]] += M2 * M2;
    O[7].outdata[Ho[t]] += Mz2;
    O[8].outdata[Ho[t]] += Mz2 * Mz2;
    O[9].outdata[Ho[t]] += Chern2;
    O[10].outdata[Ho[t]] += Chern2 * Chern2;
    //close the output of spinQ
    //O[11].outdata[Ho[t]] += spinQ1x_r * spinQ1x_r + spinQ1y_r * spinQ1y_r + spinQ1z_r * spinQ1z_r\
			    + spinQ1x_i * spinQ1x_i + spinQ1y_i * spinQ1y_i + spinQ1z_i * spinQ1z_i;
    //O[12].outdata[Ho[t]] += spinQ2x_r * spinQ2x_r + spinQ2y_r * spinQ2y_r + spinQ2z_r * spinQ2z_r\
			    + spinQ2x_i * spinQ2x_i + spinQ2y_i * spinQ2y_i + spinQ2z_i * spinQ2z_i;
    O[13].outdata[Ho[t]] += Mz;
    O[14].outdata[Ho[t]] += E*Mz;
    O[15].outdata[Ho[t]] += eta;
    O[11].outdata[Ho[t]] += (Mxx*Mxx + Myy*Myy - 2*Mxx*Myy + 4*Mxy*Mxy);
    O[12].outdata[Ho[t]] += (Mxx*Mxx + Myy*Myy - 2*Mxx*Myy + 4*Mxy*Mxy)*(Mxx*Mxx + Myy*Myy - 2*Mxx*Myy + 4*Mxy*Mxy);
    O[16].outdata[Ho[t]] += Mzi2;
    O[17].outdata[Ho[t]] += Hy;
    O[18].outdata[Ho[t]] += Iy*Iy;
    E /= H_N;
    eta /= H_N;
    if (hist_start > 0){
      if ((E<E_highest)&&(E>E_lowest)) EHistogram[Ho[t]*Slice_NUM+int(Slice_NUM*((E-E_lowest)/(E_highest-E_lowest)))] +=1;
      if ((eta<Chern_highest)&&(eta>Chern_lowest)) ChernHistogram[Ho[t]*Slice_CNUM+int(Slice_CNUM*((eta-Chern_lowest)/(Chern_highest-Chern_lowest)))] +=1;
    }
  }
}

measurement::measurement(char* indir, char* Oname, double normin, int Parallel_num){
  strcpy(name, Oname);
  strcpy(dir, indir);
  data_num = Parallel_num;
  norm = normin;
  data_mem_size = data_num * sizeof(double);
  sprintf(fn, "%s/%s", dir, name);
  outdata = (double*)calloc(data_num, sizeof(double));
}


measurement::~measurement(){
  printf("measuresingle free begin!\n");
  fflush(stdout);
  printf("measuresingle free succeed!\n");
  fflush(stdout);
}


void measurement::normalize_and_save_and_reset(){
  for (int t = 0; t < data_num; t++)
    outdata[t] = outdata[t]/norm;

  fwrite(outdata, sizeof(double), data_num, fp);

  for (int t = 0; t < data_num; t++)
    outdata[t] = 0.0;//memset????
}

void measurements::normalize_and_save_and_reset(){
  for (int i = 0; i < measurement_num; i++)
    O[i].normalize_and_save_and_reset();
}


//========================== corr part ==============================



correlation::correlation(int Pnum, char* _Corrfn){
  data_num = Pnum;
  data_num_s = Pnum/StreamN;
  Spin_mem_size_s = data_num_s * H_N * sizeof(float);
  Spin_mem_size_p_s = data_num_s * H_Nplane * sizeof(float);
  Spin_mem_size_d_s = data_num_s * H_Nplane * sizeof(double);
  Spin_mem_size_d = data_num * H_Nplane * sizeof(double);
  corrcount = 0;
  HSum = (double*)malloc(Spin_mem_size_d);
  Dcorr = (float**)calloc(StreamN, sizeof(float*));
  DSum = (double**)calloc(StreamN, sizeof(double*));
  DPo = (int**)calloc(StreamN, sizeof(int*));

  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMalloc((void**)&Dcorr[gpu_i], Spin_mem_size_p_s));
    CudaSafeCall(cudaMalloc((void**)&DSum[gpu_i], Spin_mem_size_d_s));
    CudaSafeCall(cudaMalloc((void**)&DPo[gpu_i], data_num_s * sizeof(int)));
  }

  strcpy(Corrfn, _Corrfn);
  Corrfd = open(Corrfn, O_CREAT | O_WRONLY, 0644);
  for(int i = 0; i < H_Nplane * data_num; i++){
    HSum[i] = 0.0; //initialize
  }
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpy(DSum[gpu_i], HSum + gpu_i * data_num_s * H_Nplane, Spin_mem_size_d_s, cudaMemcpyHostToDevice));
  }
}


void correlation::extract(std::vector<int>& Ho, configuration &CONF){//in &Ho[0]
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpy(DPo[gpu_i], &Ho[gpu_i * data_num_s], data_num_s * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemset(Dcorr[gpu_i], 0, Spin_mem_size_p_s));
  }
#ifndef TRI
  for (int labelx = 0; labelx < H_SpinSize; labelx += 4){
    for (int labely = 0; labely < H_SpinSize; labely += 4){
      for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
        cudaSetDevice(device_0 + gpu_i);
        GETCORR(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], Dcorr[gpu_i], labelx, labely, stream[gpu_i]);
      }
    }
  }
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    sumcorr<<<grid, block, 0, stream[gpu_i]>>>(DSum[gpu_i], Dcorr[gpu_i], DPo[gpu_i]);
  }
  CudaCheckError();
#endif
#ifdef TRI
  for (int labelx = 0; labelx < H_SpinSize; labelx += 3){
    for (int labely = 0; labely < H_SpinSize; labely += 3){
      for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
        cudaSetDevice(device_0 + gpu_i);
        GETCORR(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], Dcorr[gpu_i], labelx, labely, stream[gpu_i]);
      }
    }
  }
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    sumcorrTRI<<<grid, block, 0, stream[gpu_i]>>>(DSum[gpu_i], Dcorr[gpu_i], DPo[gpu_i]);
  }
  CudaCheckError();
#endif
  corrcount += 1;
}


void correlation::avg_write_reset(std::vector<int>& Ho){
  int gpu_i;
#ifdef TRI
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    avgcorrTRI<<<grid, block, 0, stream[gpu_i]>>>(DSum[gpu_i], double(corrcount));
  }
  CudaCheckError();
#endif
#ifndef TRI
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    avgcorr<<<grid, block, 0, stream[gpu_i]>>>(DSum[gpu_i], double(corrcount));
  }
  CudaCheckError();
#endif
	for (int j  = 0; j < data_num_s; j++){
		for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
			cudaSetDevice(device_0 + gpu_i);
			CudaSafeCall(cudaMemcpyAsync(HSum + (Ho[j + gpu_i*data_num_s]) * H_Nplane, &DSum[gpu_i][j * H_Nplane], H_Nplane * sizeof(double), cudaMemcpyDeviceToHost, stream[gpu_i]));
		}
  }
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    cudaDeviceSynchronize();
  }
  write(Corrfd, HSum, Spin_mem_size_d);
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemset(DSum[gpu_i], 0, Spin_mem_size_d_s));
  }
  corrcount = 0;
}

void correlation::changefile(char* _Corrfn){
  close(Corrfd);
  strcpy(Corrfn, _Corrfn);
  Corrfd = open(Corrfn, O_CREAT | O_WRONLY, 0644);
}

correlation::~correlation(){
  close(Corrfd);
  free(HSum);
  for (int gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaFree(this->Dcorr[gpu_i]));
    CudaSafeCall(cudaFree(this->DPo[gpu_i]));//
    CudaSafeCall(cudaFree(this->DSum[gpu_i]));
  }
}

