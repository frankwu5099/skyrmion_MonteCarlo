#ifndef PARAMS_H
#define PARAMS_H
#include "params.cuh"
#endif
#define CAL(confx, confy, confz, out) calthin<<<grid, block>>>(confx, confy, confz, out);
#define GETCORR(confx, confy, confz, corr, i, j) getcorrthin<<<grid, block>>>(confx, confy, confz, corr, i, j);

__global__ void cal2D(float *confx, float *confy, float *confz, double *out);
__global__ void calTRI(float *confx, float *confy, float *confz, double *out);
__global__ void calthin(float *confx, float *confy, float *confz, double *out);
__global__ void getcorr2D(const float *confx, const float *confy, const float *confz, float *corr, int original_i, int original_j);
__global__ void getcorrthin(const float *confx, const float *confy, const float *confz, float *corr, int original_i, int original_j);


class measurement{
  public:
    measurement(char* indir, char* Oname, int normin, int Parallel_num);
    ~measurement();
    char fn[128];
    char dir[128];
    char name[128];
    int fd;
    int data_num;
    unsigned int data_mem_size;
    double *outdata;
    double norm;

    void normalize_and_save_and_reset();
    void set_norm(int innorm);
};

class measurements{
  public:
    unsigned int Out_mem_size;
    measurements(char* indir, int Parallel_num);
    ~measurements();
    int measurement_num = 7;
    const char *names[7] = {"E", "M", "Chern", "E2", "E4", "M2", "M4"};
    const int norms[7] = {BIN_SZ * N, BIN_SZ * N, BIN_SZ,
      BIN_SZ * N * N, BIN_SZ * N * N * N * N, BIN_SZ * N * N,
      BIN_SZ * N * N * N * N};
    measurement* O;
    double *Hout;
    double *Dout;
    void measure(float* Dconfx, float* Dconfy, float* Dconfz, vector<int> Ho, float* Ms);
};

void measurements.measure(float* Dconfx, float* Dconfy, float* Dconfz, vector<int> Ho, float* Ms){
  static int raw_off;
  static double E, E2;
  static double M, My, Mz, Chern;
  CAL(Dconfx, Dconfy, Dconfz, Dout);//cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout);
  cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost);

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
    outdata[0][Ho[t]] += E;
    M2 = Mx * Mx + My * My + Mz * Mz;
    E2 = E * E;
    outdata[1][Ho[t]] += sqrt(M2);
    outdata[2][Ho[t]] += Chern;
    outdata[3][Ho[t]] += E2;
    outdata[5][Ho[t]] += M2;
    outdata[4][Ho[t]] += E2 * E2;
    outdata[6][Ho[t]] += M2 * M2;
  }
}

measurements.measurements(char * indir, int Parallel_num){
  void* raw_memmory = operator new[] (measurement_num * sizeof(measurement));
  O = static_cast<measurement*>(raw_memmory);
  for (int i =0 ; i< 7; i++){
    new (O[i])=measurement(indir, names[i], norms[i], Parallel_num);
  }
  operator delete[] (raw_memmory);
  Out_mem_size = Parallel_num * MEASURE_NUM * BN * sizeof(double);
  Hout = (double*)malloc(Out_mem_size);
  if (cudaMalloc((void**)&Dout, Out_mem_size)){
    fprintf (stderr, "Error couldn't allocate Device Data!!!!\n");
  }
}


measurements.~measurements{
  for (int i =0 ; i< 7; i++){
    O[i].~measurement();
  }
  free(Hout);
  cudaFree(Dout);
}



measuremnt.measurement(char* indir, char* Oname, int normin, int Parallel_num){
  strcpy(Oname, name);
  strcpy(indir, dir);
  data_num = Parallel_num;
  norm = normin;
  data_mem_size = data_num * sizeof(double);
  sprintf(fn, "%s/%s", dir, name);
  fd = open(fn, O_CREAT | O_WRONLY, 0644); //watch out, there might be some problem
  outdata = (double*)malloc(data_mem_size);
}


measuremnt.~measurement{
  close(fd);
}


void measurements.normalize_and_save_and_reset(){
  for (int t = 0; t < data_num; t++)
    outdata[t] = outdata[t]/norm;

  write(fd, outdata, data_mem_size);

  for (int t = 0; t < data_num; t++)
    outdata[t] = 0.0;//memset????
}


class correlation{
  correlation(int Pnum, char* dir);
  ~correlation();
  unsigned int Spin_mem_size;
  unsigned int Spin_mem_size_p;
  unsigned int Spin_mem_size_d;
  int corrcount;
  double *HSum;
  float *D;
  double *DSum;
  char Corrfn[128];
  int *DTo;//only use it only when extracting correlation 
  void avg_write_reset();//set zero
  void extract(vector<int> *Ho, configuration &CONF);//==
  //write and set
};





correlation.correlation(int Pnum, char* dir){
  data_num = Pnum;
  Spin_mem_size = Pnum * N * sizeof(float);
  Spin_mem_size_p = Pnum * Nplane * sizeof(float);
  Spin_mem_size_d = Pnum * Nplane * sizeof(double);
  corrcount = 0;
  HSum = (double*)malloc(Spin_mem_size_d);

  if(cudaMalloc((void**)&D, Spin_mem_size_p)){
    fprintf(stderr, "Error couldn't allocate Device Memory!!(Dev)\n");
    exit(1);
  }

  if(cudaMalloc((void**)&DSum, Spin_mem_size_d)){
    fprintf(stderr, "Error couldn't allocate Device Memory!!(Dsum)\n");
    exit(1);
  }
  if(cudaMalloc((void**)&DTo, Pnum * sizeof(int))){
    fprintf(stderr, "Error couldn't allocate Device Memory!!(DTo)\n");
    exit(1);
  }
  sprintf(Corrfn, "%s/Corr", dir);
  Corrfd = open(Corrfn, O_CREAT | O_WRONLY, 0644);
  for(int i = 0; i < Nplane * data_num; i++){
    HSum[i] = 0.0; //initialize
  }
  cudaMemcpy(DSum, HSum, Spin_mem_size_d, cudaMemcpyHostToDevice);
}


void correlation.extract(vector<int> *Ho, configuration &CONF){//in &Ho[0]
  cudaMemcpy(DTo, Ho, data_num * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(D, 0, Spin_mem_size);
  for (int labelx = 0; labelx < SpinSize; labelx += 4){
    for (int labely = 0; labely < SpinSize; labely += 4){
      GETCORR(CONF.Dx, CONF.Dy, CONF.Dz, D, labelx, labely);
    }
  }
  sumcorr<<<grid, block>>>(DSum, D, DTo);
  corrcount += 1;
}


void correlation.avg_write_reset(){
  avgcorr<<<grid, block>>>(DSum, double(corrcount));
  cudaMemcpy(HSum, DSum, Spin_mem_size_d, cudaMemcpyDeviceToHost);
  write(Corrfd, HSum, Spin_mem_size_d);
  cudaMemset((void*)DSum, 0, Spin_mem_size_d);
}



void correlation.~correlation(){
  close(Corrfd);
  free(HSum);
  cudaFree(D);
  cudaFree(DTo);//
  cudaFree(DSum);
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

