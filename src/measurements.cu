#include "measurements.cuh"


measurements::measurements(char * indir, int Parallel_num, unsigned int binSize){
  measurement_num = 7;
  void* raw_memmory = operator new[] (measurement_num * sizeof(measurement));
  strcpy(names[0], "E");
  strcpy(names[1], "M");
  strcpy(names[2], "Chern");
  strcpy(names[3], "E2");
  strcpy(names[4], "E4");
  strcpy(names[5], "M2");
  strcpy(names[6], "M4");
  norms[0] = binSize * N;
  norms[1] = binSize * N;
  norms[2] = binSize;
  norms[3] = binSize * N * N;
  norms[4] = binSize * N * N * N * N;
  norms[5] = binSize * N * N;
  norms[6] = binSize * N * N * N * N;
  O = static_cast<measurement*>(raw_memmory);
  for (int i =0 ; i< measurement_num; i++){
    new (&O[i])measurement(indir, names[i], norms[i], Parallel_num);
  }
  data_num = Parallel_num;
  Out_mem_size = Parallel_num * MEASURE_NUM * BN * sizeof(double);
  printf("%u\n", Out_mem_size);
  Hout = (double*)malloc(Out_mem_size);
  CudaSafeCall(cudaMalloc(&Dout, Out_mem_size));
  operator delete[] (raw_memmory);
}


measurements::~measurements(){
  for (int i =0 ; i< 7; i++){
    O[i].~measurement();
  }
  free(Hout);
  CudaSafeCall(cudaFree(Dout));
}



void measurements::measure(float* Dconfx, float* Dconfy, float* Dconfz, std::vector<int>& Ho, double* Ms, float* HHs){
  static int raw_off;
  static double E, E2;
  static double Mx, My, Mz, Chern, M2;
  CAL(Dconfx, Dconfy, Dconfz, Dout);//cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost));

  for(int t = 0; t < data_num; t++){
    raw_off = t * MEASURE_NUM * BN;
    E = 0, E2 = 0;
    Mx = 0, My = 0, Mz = 0, Chern = 0;
    for(int j = 0; j < BN; j++)
      E += Hout[raw_off + j];
    for(int j = BN; j < 2 * BN; j++)
      Mx += Hout[raw_off + j];
    for(int j = 2 * BN; j < 3 * BN; j++)
      My += Hout[raw_off + j];
    for(int j = 3 * BN; j < 4 * BN; j++)
      Mz += Hout[raw_off + j];
    for(int j = 4 * BN; j < 5 * BN; j++)
      Chern += Hout[raw_off + j];
    Ms[Ho[t]] = Mz;	//Es is the energies in order of temperature set
    E = E - HHs[t] * Mz;
    O[0].outdata[Ho[t]] += E;
    M2 = Mx * Mx + My * My + Mz * Mz;
    E2 = E * E;
    O[1].outdata[Ho[t]] += sqrt(M2);
    O[2].outdata[Ho[t]] += Chern;
    O[3].outdata[Ho[t]] += E2;
    O[5].outdata[Ho[t]] += M2;
    O[4].outdata[Ho[t]] += E2 * E2;
    O[6].outdata[Ho[t]] += M2 * M2;
  }
}

measurement::measurement(char* indir, char* Oname, int normin, int Parallel_num){
  strcpy(Oname, name);
  strcpy(indir, dir);
  data_num = Parallel_num;
  norm = normin;
  data_mem_size = data_num * sizeof(double);
  sprintf(fn, "%s/%s", dir, name);
  fd = open(fn, O_CREAT | O_WRONLY, 0644); //watch out, there might be some problem
  outdata = (double*)calloc(data_num, sizeof(double));
}


measurement::~measurement(){
  close(fd);
}


void measurement::normalize_and_save_and_reset(){
  for (int t = 0; t < data_num; t++)
    outdata[t] = outdata[t]/norm;

  write(fd, outdata, data_mem_size);

  for (int t = 0; t < data_num; t++)
    outdata[t] = 0.0;//memset????
}

void measurements::normalize_and_save_and_reset(){
  for (int i = 0; i < measurement_num; i++)
    O[i].normalize_and_save_and_reset();
}


//========================== corr part ==============================



correlation::correlation(int Pnum, char* dir){
  data_num = Pnum;
  Spin_mem_size = Pnum * N * sizeof(float);
  Spin_mem_size_p = Pnum * Nplane * sizeof(float);
  Spin_mem_size_d = Pnum * Nplane * sizeof(double);
  corrcount = 0;
  HSum = (double*)malloc(Spin_mem_size_d);

  CudaSafeCall(cudaMalloc((void**)&D, Spin_mem_size_p));

  CudaSafeCall(cudaMalloc((void**)&DSum, Spin_mem_size_d));
  CudaSafeCall(cudaMalloc((void**)&DPo, Pnum * sizeof(int)));
  sprintf(Corrfn, "%s/Corr", dir);
  Corrfd = open(Corrfn, O_CREAT | O_WRONLY, 0644);
  for(int i = 0; i < Nplane * data_num; i++){
    HSum[i] = 0.0; //initialize
  }
  CudaSafeCall(cudaMemcpy(DSum, HSum, Spin_mem_size_d, cudaMemcpyHostToDevice));
}


void correlation::extract(std::vector<int>* Ho, configuration &CONF){//in &Ho[0]
  CudaSafeCall(cudaMemcpy(DPo, Ho, data_num * sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemset(D, 0, Spin_mem_size));
#ifndef TRI
  for (int labelx = 0; labelx < SpinSize; labelx += 4){
    for (int labely = 0; labely < SpinSize; labely += 4){
      GETCORR(CONF.Dx, CONF.Dy, CONF.Dz, D, labelx, labely);
    }
  }
  sumcorr<<<grid, block>>>(DSum, D, DPo);
  CudaCheckError();
#endif
#ifdef TRI
  for (int labelx = 0; labelx < SpinSize; labelx += 3){
    for (int labely = 0; labely < SpinSize; labely += 3){
      GETCORR(CONF.Dx, CONF.Dy, CONF.Dz, D, labelx, labely);
    }
  }
  sumcorrTRI<<<grid, block>>>(DSum, D, DPo);
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
  CudaSafeCall(cudaMemset((void*)DSum, 0, Spin_mem_size_d));
}



correlation::~correlation(){
  close(Corrfd);
  free(HSum);
  CudaSafeCall(cudaFree(D));
  CudaSafeCall(cudaFree(DPo));//
  CudaSafeCall(cudaFree(DSum));
}
/*
   char Efn[128];
   char Mfn[128];
   char Chernfn[128];
   char E2fn[128];
   char E4fn[128];
   char M2fn[128];
   char M4fn[128];
   sprintf(Efn, "%s/E", dir);
   sprintf(Mfn, "%s/M", dir);
   sprintf(Chernfn, "%s/Chern", dir);
   sprintf(E2fn, "%s/E2", dir);
   sprintf(E4fn, "%s/E4", dir);
   sprintf(M2fn, "%s/M2", dir);
   sprintf(M4fn, "%s/M4", dir);
   int Efd = open(Efn, O_CREAT | O_WRONLY, 0644);
   int Mfd = open(Mfn, O_CREAT | O_WRONLY, 0644);
   int Chernfd = open(Chernfn, O_CREAT | O_WRONLY, 0644);
   int E2fd = open(E2fn, O_CREAT | O_WRONLY, 0644);
   int E4fd = open(E4fn, O_CREAT | O_WRONLY, 0644);
   int M2fd = open(M2fn, O_CREAT | O_WRONLY, 0644);
   int M4fd = open(M4fn, O_CREAT | O_WRONLY, 0644);
   double *Eout = (double*)malloc(data_mem_size);
   double *Mout = (double*)malloc(data_mem_size);
   double *Chernout = (double*)malloc(data_mem_size);
   double *E2out = (double*)malloc(data_mem_size);
   double *E4out = (double*)malloc(data_mem_size);
   double *M2out = (double*)malloc(data_mem_size);
   double *M4out = (double*)malloc(data_mem_size);
   double *binE = (double*)calloc(Hnum, sizeof(double));
   double *binM = (double*)calloc(Hnum, sizeof(double));
   double *binChern = (double*)calloc(Hnum, sizeof(double));
   double *binE2 = (double*)calloc(Hnum, sizeof(double));
   double *binM2 = (double*)calloc(Hnum, sizeof(double));
   double *binE4 = (double*)calloc(Hnum, sizeof(double));
   double *binM4 = (double*)calloc(Hnum, sizeof(double));
   for(int t = 0; t < Hnum; t++){
   Eout[t] = (double)binE[t] / BIN_SZ / N;
   Mout[t] = (double)binM[t] / BIN_SZ / N;
   Chernout[t] = (double)binChern[t] / BIN_SZ;
   E2out[t] = (double)binE2[t] / BIN_SZ / N / N;
   E4out[t] = (double)binE4[t] / BIN_SZ / N / N / N / N;
   M2out[t] = (double)binM2[t] / BIN_SZ / N / N;
   M4out[t] = (double)binM4[t] / BIN_SZ / N / N / N / N;
//Helout[t] = (double)binHel[t] / BIN_SZ / N;
}
write(Efd, Eout, data_mem_size);
write(Mfd, Mout, data_mem_size);
write(Chernfd, Chernout, data_mem_size);
write(E2fd, E2out, data_mem_size);
write(E4fd, E4out, data_mem_size);
write(M2fd, M2out, data_mem_size);
write(M4fd, M4out, data_mem_size);
close(Efd);
close(Mfd);
close(Chernfd);
close(E2fd);
close(E4fd);
close(M2fd);
close(M4fd);
free(Eout);
free(Mout);
free(Chernout);
free(E2out);
free(E4out);
free(M2out);
free(M4out);
 */

