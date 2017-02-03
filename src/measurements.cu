#include "measurements.cuh"


measurements::measurements(char * indir, int Parallel_num, unsigned int binSize){
  measurement_num = 14;
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
  strcpy(names[11], "SQ1");
  strcpy(names[12], "SQ2");
  strcpy(names[13], "Mz");
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
  norms[12] = double(binSize) * H_N * H_N;
  norms[13] = double(binSize) * H_N;
  O.reserve(measurement_num);
  for (int i =0 ; i< measurement_num; i++){
    O.push_back(measurement(indir, names[i], norms[i], Parallel_num));
    O[O.size() - 1].fp = fopen(O[O.size() - 1].fn, "w");
  }
  data_num = Parallel_num;
  Out_mem_size = Parallel_num * MEASURE_NUM * H_BN * sizeof(double);
  printf("%u\n", Out_mem_size);
  Hout = (double*)malloc(Out_mem_size);
  CudaSafeCall(cudaMalloc(&Dout, Out_mem_size));
  EHistogram = (unsigned int*) calloc(Parallel_num * Slice_NUM, sizeof(unsigned int));
  ChernHistogram = (unsigned int*) calloc(Parallel_num * Slice_CNUM, sizeof(unsigned int));
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



void measurements::virtual_measure(float* Dconfx, float* Dconfy, float* Dconfz, std::vector<int>& Ho, double* Ms, double* Es, float* HHs){
  static int raw_off;
  static double E;
  static double Mz;
  CAL(Dconfx, Dconfy, Dconfz, Dout);//cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost));

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



void measurements::measure(float* Dconfx, float* Dconfy, float* Dconfz, std::vector<int>& Ho, double* Ms, double* Es, float* HHs){
  static int raw_off;
  static double E, E2;
  static double Mx, My, Mz, Chern, M2, Mz2, Chern2;
  static double spinQ1x_r, spinQ1y_r, spinQ1z_r, spinQ1x_i, spinQ1y_i, spinQ1z_i;
  static double spinQ2x_r, spinQ2y_r, spinQ2z_r, spinQ2x_i, spinQ2y_i, spinQ2z_i;
  CAL(Dconfx, Dconfy, Dconfz, Dout);//cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost));

  for(int t = 0; t < data_num; t++){
    raw_off = t * MEASURE_NUM * H_BN;
    E = 0, E2 = 0;
    Mx = 0, My = 0, Mz = 0, Chern = 0;
    spinQ1x_r = 0, spinQ1y_r = 0, spinQ1z_r = 0;
    spinQ1x_i = 0, spinQ1y_i = 0, spinQ1z_i = 0;
    spinQ2x_r = 0, spinQ2y_r = 0, spinQ2z_r = 0;
    spinQ2x_i = 0, spinQ2y_i = 0, spinQ2z_i = 0;
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
    Ms[Ho[t]] = Mz;	//Es is the energies in order of temperature set
    E = E - HHs[t] * Mz;
    Es[Ho[t]] = E;	//Es is the energies in order of temperature set
    O[0].outdata[Ho[t]] += E;
    M2 = Mx * Mx + My * My + Mz * Mz;
    Mz2 = Mz * Mz;
    Chern2 = Chern * Chern;
    E2 = E * E;
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
    O[11].outdata[Ho[t]] += spinQ1x_r * spinQ1x_r + spinQ1y_r * spinQ1y_r + spinQ1z_r * spinQ1z_r\
			    + spinQ1x_i * spinQ1x_i + spinQ1y_i * spinQ1y_i + spinQ1z_i * spinQ1z_i;
    O[12].outdata[Ho[t]] += spinQ2x_r * spinQ2x_r + spinQ2y_r * spinQ2y_r + spinQ2z_r * spinQ2z_r\
			    + spinQ2x_i * spinQ2x_i + spinQ2y_i * spinQ2y_i + spinQ2z_i * spinQ2z_i;
    O[13].outdata[Ho[t]] += Mz;
    E /= H_N;
    if ((E<E_highest)&&(E>E_lowest)) EHistogram[Ho[t]*Slice_NUM+int(Slice_NUM*((E-E_lowest)/(E_highest-E_lowest)))] +=1;
    if ((-Chern<Chern_highest)&&(-Chern>Chern_lowest)) ChernHistogram[Ho[t]*Slice_CNUM+int(Slice_CNUM*((Chern-Chern_lowest)/(Chern_highest-Chern_lowest)))] +=1;
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
  Spin_mem_size = Pnum * H_N * sizeof(float);
  Spin_mem_size_p = Pnum * H_Nplane * sizeof(float);
  Spin_mem_size_d = Pnum * H_Nplane * sizeof(double);
  corrcount = 0;
  HSum = (double*)malloc(Spin_mem_size_d);

  CudaSafeCall(cudaMalloc((void**)&Dcorr, Spin_mem_size_p));
  CudaSafeCall(cudaMalloc((void**)&skyr_den, Spin_mem_size_p));

  CudaSafeCall(cudaMalloc((void**)&DSum, Spin_mem_size_d));
  CudaSafeCall(cudaMalloc((void**)&DPo, Pnum * sizeof(int)));
  strcpy(Corrfn, _Corrfn);
  Corrfd = open(Corrfn, O_CREAT | O_WRONLY, 0644);
  for(int i = 0; i < H_Nplane * data_num; i++){
    HSum[i] = 0.0; //initialize
  }
  CudaSafeCall(cudaMemcpy(DSum, HSum, Spin_mem_size_d, cudaMemcpyHostToDevice));
}


void correlation::extract(std::vector<int>& Ho, configuration &CONF){//in &Ho[0]
  CudaSafeCall(cudaMemcpy(DPo, &Ho[0], data_num * sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemset(Dcorr, 0, Spin_mem_size_p));
#ifndef TRI
  for (int labelx = 0; labelx < H_SpinSize; labelx += 4){
    for (int labely = 0; labely < H_SpinSize; labely += 4){
      GETCORR(CONF.Dx, CONF.Dy, CONF.Dz, Dcorr, labelx, labely);
    }
  }
  sumcorr<<<grid, block>>>(DSum, Dcorr, DPo);
  CudaCheckError();
#endif
#ifdef TRI

  GETSKYRDEN(CONF.Dx, CONF.Dy, CONF.Dz, skyr_den);
  for (int labelx = 0; labelx < H_SpinSize; labelx += 3){
    for (int labely = 0; labely < H_SpinSize; labely += 3){
      GETCORR(skyr_den, Dcorr, labelx, labely);
    }
  }
  sumcorrTRI<<<grid, block>>>(DSum, Dcorr, DPo);
  CudaCheckError();
#endif
  corrcount += 1;
}


void correlation::avg_write_reset(){
#ifdef TRI
  avgcorrTRI<<<grid, block>>>(DSum, double(corrcount));
  CudaCheckError();
#endif
#ifndef TRI
  avgcorr<<<grid, block>>>(DSum, double(corrcount));
  CudaCheckError();
#endif
  CudaSafeCall(cudaMemcpy(HSum, DSum, Spin_mem_size_d, cudaMemcpyDeviceToHost));
  write(Corrfd, HSum, Spin_mem_size_d);
  CudaSafeCall(cudaMemset(DSum, 0, Spin_mem_size_d));
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
  CudaSafeCall(cudaFree(this->Dcorr));
  CudaSafeCall(cudaFree(this->skyr_den));
  CudaSafeCall(cudaFree(this->DPo));//
  CudaSafeCall(cudaFree(this->DSum));
}

