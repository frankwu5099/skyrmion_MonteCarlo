//Ms PT function writedata  HHs
//why exchange conf and measurement works
using namespace std;

#define BIN_SZ 3000//0//00//
#define BIN_NUM 3//0
#define EQUI_N 20000//0//0//00////16000000

#define ID "skyr_d16AC_testmeasurement_TRI"
#define PTF	(float(0.00))	//Frequency of parallel tempering
#include "params.cuh"
#include "updates.cuh"
#include "measurements.cuh"
#include "configuration.cuh"
#include "extend.cu"
#define EQUI_Ni (4000)//0)
#define GET_CORR
#define f_CORR (500)


unsigned seed = 73;
mt19937 rng(seed);
uniform_01<mt19937> uni01_sampler(rng);
void tempering(double*, int*);
unsigned int block = BlockSize_x * BlockSize_y;
unsigned int grid = 0;
vector<float> Tls;
vector<float> Hls;
vector<int>Po;		//order of Temperature, Tls[To[t]] is the temperature of t'th configuration.
vector<int>ivPo;		//order of Temperature, Tls[To[t]] is the temperature of t'th configuration.
unsigned int Tnum;
unsigned int Hnum;
float Cparameter = 0.8;
void var_examine();



int main(int argc, char *argv[]){
  //call GPU

  if (setDev()==1){
    return 1;
  }
  cudaGetLastError();
  CudaCheckError();

  //examine variables
  var_examine();

  // ========================== initialize ===========================

  //begin (read in temperatures)
  unsigned int Temp_mem_size;
  unsigned int H_mem_size;

  if(argc > 2){
    float tmp;
    FILE *Tfp = fopen(argv[1], "r");
    int i = 0;
    while(fscanf(Tfp, "%f", &tmp) != EOF){
      Tls.push_back(tmp);
      i++;
    }
    fclose(Tfp);
    Tnum = Tls.size();
    Temp_mem_size = Tnum * sizeof(float);

    i = 0;
    Tfp = fopen(argv[2], "r");
    while(fscanf(Tfp, "%f", &tmp) != EOF){
      Hls.push_back((DD * DD + DR * DR)*tmp);
      i++;
    }
    fclose(Tfp);
    Hnum = Hls.size();
    H_mem_size = Hnum * sizeof(float);

    grid = Pnum * BN;
  }
  else{
    fprintf(stderr, "Give me a temperature set!!!\n");
    fprintf(stderr, "Give me a field set!!!\n");
    exit(0);
  }
  //end (read in temperatures)


  //invTs is the inverse temperature in order of configurations on GPU.
  for (int i = 0; i < Pnum ; i++){
    Po.push_back(i);
    ivPo.push_back(i);
  }
  float *Cparameters;
  float *HPparameters;
  float *DPparameters;
  Cparameters = (float*)malloc(C_mem_size);
  HPparameters = (float*)malloc(P_mem_size);
  CudaSafeCall(cudaMalloc((void**)&DPparameters, P_mem_size));


  //begin (initialize random seeds)
  //Declare sizes
  unsigned int totalRngs = Pnum * TN / WarpStandard_K;
  unsigned seedBytes = totalRngs * sizeof(unsigned int) * WarpStandard_STATE_WORDS;
  unsigned int *seedDevice = 0;
  CudaSafeCall(cudaMalloc((void **)&seedDevice, seedBytes));
  unsigned int* seedHost = (unsigned int*)malloc(seedBytes);
  srand(seed);
  for(int i = 0; i < seedBytes / sizeof(unsigned int); i++)
    seedHost[i] = uni01_sampler() * UINT_MAX;
  CudaSafeCall(cudaMemcpy(seedDevice, seedHost, seedBytes, cudaMemcpyHostToDevice));
  //end (initialize random seeds)


  //Set up output data path
  char dir[128];
  sprintf(dir, "Data/L_%d-%s", SpinSize, ID);
  mkdir(dir, 0755);
  char Seedfn[128];
  sprintf(Seedfn, "Data/L_%d-%s/seed", SpinSize, ID);
  int seedfd = open(Seedfn, O_CREAT | O_WRONLY, 0644);
  write(seedfd, seedHost, seedBytes);
  close(seedfd);


  //MEASUREMENT initialize

  configuration CONF(Pnum, dir);
  measurements MEASURE(dir, Pnum, BIN_SZ); //Tnum for parallel tempering for T
#ifdef GET_CORR
  correlation CORR(Pnum, dir);
#endif

  StopWatchInterface *timer=NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);

  //Give initial configuration and settle the systems down to equilibrium states
  CONF.initialize(ORDER);
  int Eqii = 0;//150;
  for(int i = 0; i < Hnum; i++)
    HHs[i] = Hls[i];
  for(int i = 0; i < Tnum; i++)
    invTs[i] = 1.0/Tls[i];
  CudaSafeCall(cudaMemcpy(DPparameters, HPparameters, P_mem_size, cudaMemcpyHostToDevice));
  double *Ms = (double*)malloc(Pnum * sizeof(double));
  int *accept1 = (int*)calloc(Pnum - 1, sizeof(int));
  float cnt = 0;
  Cparameter = Cparameters[Cnum-1];

  for(int i = 0; i < EQUI_N; i++){
    if (i % 10 ==0) printf("%d\n",i);
    SSF(CONF.Dx, CONF.Dy, CONF.Dz, seedDevice, DPparameters, Cparameter);
    //================================= no PT ==========================================
    /*
       cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout); //, Dcorr);
       cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost);
       for(int t = 0; t < Pnum; t++){
       int raw_off = t * MEASURE_NUM * BN;
       E = 0;
       for(int j = 3 * BN; j < 4 * BN; j++)
       E += Hout[raw_off + j];
       Ms[Ho[t]] = E;	//Es is the energies in order of temperature set
       }
    //Parallel Tempering
    cnt += PTF;
    for(int p = 0; p < int(cnt); p++){
    tempering(Ms, accept1);
    for(int t = 0; t < Pnum; t++) HPparameters[t] = Porder(t);
    cudaMemcpy(DPparameters, HPparameters, H_mem_size, cudaMemcpyHostToDevice);
    }
    if(int(cnt))
    cnt = 0;
     */
  }

  //Do measurements (annealing)

  int *accept = (int*)calloc(Pnum - 1, sizeof(int));
  for(int C_i = 0 ; C_i < Cnum ; C_i ++){
    for (int i = 0; i< Pnum-1; i++) accept[i] = 0;
    Cparameter = Ccurrent(C_i);
    for(int i = 0; i < EQUI_Ni; i++){
      if (i % 10 ==0) printf("%f : %d\n", Cparameter, i);
      SSF(CONF.Dx, CONF.Dy, CONF.Dz, seedDevice, DPparameters, Cparameter);
      //======================= no PT ===============================
      /*
	 cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout); //, Dcorr);
	 cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost);
	 for(int t = 0; t < Pnum; t++){
	 int raw_off = t * MEASURE_NUM * BN;
	 E = 0;
	 for(int j = 3 * BN; j < 4 * BN; j++)
	 E += Hout[raw_off + j];
	 Ms[Ho[t]] = E;	//Es is the energies in order of temperature set
	 }
      //Parallel Tempering
      cnt += PTF;
      for(int p = 0; p < int(cnt); p++){
      tempering(Ms, accept1);
      for(int t = 0; t < Pnum; t++) HPparameters[t] = Porder(t);
      cudaMemcpy(DPparameters, HPparameters, P_mem_size, cudaMemcpyHostToDevice);
      }
      if(int(cnt))
      cnt = 0;
       */
    }
    cnt = 0;
    for(int b = 0; b < BIN_NUM; b++){
      //Take the ensemble average
      for(int i = 0; i < BIN_SZ; i++){
	SSF(CONF.Dx, CONF.Dy, CONF.Dz, seedDevice, DPparameters, Cparameter);
	MEASURE.measure(CONF.Dx, CONF.Dy, CONF.Dz, Po, Ms, HHs);
#ifdef GET_CORR
	if ( i % f_CORR==0){
	  CORR.extract(Po, CONF);//==
	}
#endif
	//Parallel Tempering
	/*
	   cnt += PTF;
	   for(int p = 0; p < int(cnt); p++){
	   tempering(Ms, accept);
	   for(int t = 0; t < Pnum; t++) HPparameters[t] = Porder(t);
	   cudaMemcpy(DPparameters, HPparameters, P_mem_size, cudaMemcpyHostToDevice);
	   }
	   if(int(cnt))
	   cnt = 0;
	 */
      }
      MEASURE.normalize_and_save_and_reset();
    }
    for (int iii = 0 ; iii < Pnum; iii ++){
      ivPo[Po[iii]] = Po[iii];
    }
    CONF.backtoHost();
    CONF.writedata();
#ifdef GET_CORR
    CORR.avg_write_reset();
#endif
  }
  free(Ms);
  sdkStopTimer(&timer);
  double time = 1.0e-3 * sdkGetTimerValue(&timer);


//======================= print details ==========================
  char detailFn[128];
  sprintf(detailFn, "%s/details", dir);
  FILE *detailFp = fopen(detailFn, "w");
  fprintf(detailFp, "elapsed time = %f (sec)\n", time);
  double speed = 0;
  speed = (N / time / 1000000000) * (BIN_SZ * BIN_NUM + EQUI_N) * Pnum * Cnum;
  fprintf(detailFp, "speed = %f (GHz)\n", speed);
  fprintf(detailFp, "RNG: WarpStandard\n", SpinSize);
  fprintf(detailFp, "SpinSize = %d\n", SpinSize);
  fprintf(detailFp, "A = %4.3f\n", A);
  fprintf(detailFp, "D_Rashba = %4.3f\n", DR);
  fprintf(detailFp, "D_Dresselhaus = %4.3f\n", DD);
  fprintf(detailFp, "BlockSize_x = %d\n", BlockSize_x);
  fprintf(detailFp, "BlockSize_y = %d\n", BlockSize_y);
  fprintf(detailFp, "GridSize_x = %d\n", GridSize_x);
  fprintf(detailFp, "GridSize_y = %d\n", GridSize_y);
  fprintf(detailFp, "Bin Size = %d\n", BIN_SZ);
  fprintf(detailFp, "Bin Number = %d\n", BIN_NUM);
  fprintf(detailFp, "Equilibration N = %d\n", EQUI_N);
  fprintf(detailFp, "Equilibration Ni = %d\n", EQUI_Ni);
  fprintf(detailFp, "f_CORR = %d\n", f_CORR);
  fprintf(detailFp, "PT frequency = %3.2f\n", PTF);
  fprintf(detailFp, "Tnum = %d\n", Tnum);
  fprintf(detailFp, "Temperature Set: ");
  for(int i = 0; i < Tnum; i++){
    fprintf(detailFp, "%.5f  ", Tls[i]);
  }
  fprintf(detailFp, "\nHnum = %d\n", Hnum);
  fprintf(detailFp, "field Set: ");
  for(int i = 0; i < Hnum; i++){
    fprintf(detailFp, "%.5f  ", Hls[i]/(DR*DR + DD*DD));
  }
  for(int i = 0; i < Pnum; i++){
    fprintf(detailFp, "\n");
    fprintf(detailFp, "Po[%d]=%d",i,Po[i]);
  }
  fprintf(detailFp, "\n");
  fprintf(detailFp, "Acceptance rates: ");
  if (PTF != 0 ){
    for(int i = 0; i < Pnum - 1; i++)
      fprintf(detailFp, "%4.3f  ", float(accept[i]) / (BIN_SZ * BIN_NUM * PTF));
  }
  fprintf(detailFp, "\n");
  if (ORDER){
    fprintf(detailFp, "Configurations start from ordered state.\n");
  }
  else {
    fprintf(detailFp, "Configurations start from random state.\n");
  }
  fprintf(detailFp, "Done by Po-Kuan Wu ^_^\n", EQUI_N);
  fclose(detailFp);
//===================== print details end =========================

  //Set free memory
  free(seedHost);
  CudaSafeCall(cudaFree(DPparameters));
  CudaSafeCall(cudaFree(seedDevice));
  //CORR.~correlation();
  //MEASURE.~measurements();
  //CONF.~configuration();
  return 0;
}


//=============================== functions ==================================
void tempering(double *Ms, int *accept){
  int map[Pnum];	//map[t] the configuration of t'th temperature
  for(int i = 0; i < Pnum; i++){
    map[Po[i]] = i;
  }
  double delta;
  int flag = 0;
  for(int i = 0; i < Pnum - 1; i++){
    delta = exchangecriterion(i);
    if(delta > 0)
      flag = 1;
    else if(uni01_sampler() < exp(delta))
      flag = 1;
    if(flag){
      int tmp = Po[map[i]];
      Po[map[i]] = Po[map[i + 1]];
      Po[map[i + 1]] = tmp;
      tmp = map[i];
      map[i] = map[i + 1];
      map[i + 1] = tmp;
      double tmpE = Ms[i];
      Ms[i] = Ms[i + 1];
      Ms[i + 1] = tmpE;
      accept[i] += 1;
      flag = 0;
    }
  }
}


void var_examine(){
#ifndef TRI
  if(SpinSize % (BlockSize_x * 2) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", BlockSize_x * 2);
    exit(0);
  }
  if(SpinSize % (BlockSize_y * 2) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", BlockSize_y * 2);
    exit(0);
  }
#endif
#ifdef TRI
  if(SpinSize % (BlockSize_x * 3) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", BlockSize_x * 2);
    exit(0);
  }
  if(SpinSize % (BlockSize_y * 3) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", BlockSize_y * 2);
    exit(0);
  }
#endif
#ifndef THIN
  if (SpinSize_z != 1){
    fprintf(stderr, "SpinSize_z must be 1 %d\n", BlockSize_y * 2);
    exit(0);
  }
#endif
}
