//Ms PT function writedata  HHs
//Po -> Cpara
//why exchange conf and measurement works
using namespace std;


#include "params.cuh"
#include "updates.cuh"
#include "configuration.cuh"
#include "measurements_sk.cuh"
#include "extend.cu"
#define GET_CORR


unsigned seed = 73;
mt19937 rng(seed);
uniform_01<mt19937> uni01_sampler(rng);
void tempering_simple(double*, double*, int*);
void tempering(double*, double*, int*, int*, int*);
vector< vector<float> > Tls;
vector< vector<float> > Hls;
vector<int> Po;		//order of Temperature, Tls[To[t]] is the temperature of t'th configuration.
vector<int> ivPo;		//order of Temperature, Tls[To[t]] is the temperature of t'th configuration.
unsigned int Pnum;
unsigned int Cnum;
unsigned int Tnum;
unsigned int Hnum;
unsigned int f_CORR;
unsigned int CORR_N;
float Cparameter = 0.8;
int C_i = 0;
void var_examine();



int main(int argc, char *argv[]){
  //call GPU
  read_params(argv[1]);
  int deviceNum, gpu_i;
  cudaGetDeviceCount(&deviceNum);
  device_0 = setDev();
  StreamN = deviceNum -device_0;

  if (device_0 == -1){
    return 1;
  }
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    cudaStreamCreate(&stream[gpu_i]);
  }
  cudaGetLastError();
  CudaCheckError();
  //note: gpu part

  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    move_params_device_flip();
    move_params_device_cals();
    move_params_device_corr();
  }

  //examine variables
  var_examine();

  // ========================== initialize ===========================

  //begin (read in temperatures)
  unsigned int params_mem_size;

  if(argc > 2){
    float tmpT, tmpH;
    FILE *paramsfp = fopen(argv[2], "r");
    vector<float> tmpTls;
    vector<float> tmpHls;
    fscanf(paramsfp, "%d %d", &Tnum, &Hnum);
    while(fscanf(paramsfp, "%f %f", &tmpT, &tmpH) != EOF){
      if (tmpT < 0){
        Tls.push_back(tmpTls);
        Hls.push_back(tmpHls);
        tmpTls.clear();
        tmpHls.clear();
      }
      else {
        tmpTls.push_back(tmpT);
        tmpHls.push_back(tmpH*(DR*DR + DD*DD));
      }
    }
    fclose(paramsfp);
    Pnum = Tls[0].size();
    Cnum = Tls.size();
    if (Tnum * Hnum != Pnum){
      fprintf(stderr, "wrong temperatures and fields!!!\n");
      exit(0);
    }
    params_mem_size = Pnum * sizeof(float);

  }
  else{
    fprintf(stderr, "Give me a temperature set!!!\n");
    fprintf(stderr, "Give me a field set!!!\n");
    exit(0);
  }
  //end (read in temperatures)

  if (Pnum%StreamN != 0){
    printf("Fatal error: The number of replicas is not consistent with the number of streams.");
    return 1;
  }
  int Pnum_s = Pnum/StreamN;
  int params_mem_size_s = Pnum_s * sizeof(float);
	grid = Pnum_s * H_BN;


  //invTs is the inverse temperature in order of configurations on GPU.
  for (int i = 0; i < Pnum ; i++){
    Po.push_back(i);
    ivPo.push_back(i);
  }
  float *HHs;
  float **DHs;
  DHs = (float**)calloc(StreamN, sizeof(float*));
  float *invTs;
  float **DinvTs;
  DinvTs = (float**)calloc(StreamN, sizeof(float*));
  HHs = (float*)malloc(params_mem_size);
  invTs = (float*)malloc(params_mem_size);

  //note: gpu part
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMalloc((void**)&DHs[gpu_i], params_mem_size/StreamN));
    CudaSafeCall(cudaMalloc((void**)&DinvTs[gpu_i], params_mem_size/StreamN));
  }

  //begin (initialize random seeds)
  //Declare sizes
  unsigned int totalRngs = Pnum * H_BN * H_TN / WarpStandard_K;
  unsigned seedBytes = totalRngs * sizeof(unsigned int) * WarpStandard_STATE_WORDS;
  unsigned int **seedDevice;
  seedDevice = (unsigned int**)calloc(StreamN, sizeof(unsigned int*));
  //note: gpu part
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMalloc((void **)&seedDevice[gpu_i], seedBytes/StreamN));
  }
  unsigned int* seedHost = (unsigned int*)malloc(seedBytes);
  srand(seed);
  for(int i = 0; i < seedBytes / sizeof(unsigned int); i++)
    seedHost[i] = uni01_sampler() * UINT_MAX;
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyAsync(seedDevice[gpu_i], seedHost + (seedBytes/sizeof(unsigned int)/StreamN)*gpu_i, (seedBytes/StreamN), cudaMemcpyHostToDevice, stream[gpu_i]));
  }
  //end (initialize random seeds)


  //Set up output data path
  char dir[128];
  sprintf(dir, "Data/L_%d-%s", H_SpinSize, Output);
  mkdir(dir, 0755);
  char Seedfn[128];
  sprintf(Seedfn, "Data/L_%d-%s/seed", H_SpinSize, Output);
  int seedfd = open(Seedfn, O_CREAT | O_WRONLY, 0644);
  write(seedfd, seedHost, seedBytes);
  close(seedfd);


  //MEASUREMENT initialize

  configuration CONF(Pnum, dir);
  measurements MEASURE(dir, Pnum, BIN_SZ); //Tnum for parallel tempering for T
#ifdef GET_CORR
  char Corrfn[128];
  sprintf(Corrfn, "%s/Corrpt", dir);
  correlation CORR(Pnum, Corrfn);
#endif

  StopWatchInterface *timer=NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);

  //Give initial configuration and settle the systems down to equilibrium states
  CONF.initialize(ORDER);
  int Eqii = 0;//150;
  for(int i = 0; i < Pnum; i++){
    HHs[i] = Hls[0][i];
    invTs[i] = 1.0/Tls[0][i];
  }
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaMemcpyAsync(DinvTs[gpu_i], invTs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
    CudaSafeCall(cudaMemcpyAsync(DHs[gpu_i], HHs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
  }
  double *Ms = (double*)malloc(Pnum * sizeof(double));
  double *Es = (double*)malloc(Pnum * sizeof(double));
  int *accept1 = (int*)calloc((Tnum - 1)*Hnum + Tnum*(Hnum - 1), sizeof(int));
  int *stay = (int*)calloc(Tnum * Hnum, sizeof(int));
  int *staylargest = (int*)calloc(Tnum * Hnum, sizeof(int));
  int *staytmp = (int*)calloc(Tnum * Hnum, sizeof(int));
  float cnt = 0;

  for(int i = 0; i < EQUI_N; i++){
    if (i % 10 ==0) printf("%d\n",i);
    for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
      cudaSetDevice(device_0 + gpu_i);
      SSF1(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
    }
    for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
      cudaSetDevice(device_0 + gpu_i);
      SSF2(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
    }
    for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
      cudaSetDevice(device_0 + gpu_i);
      SSF3(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
    }
    //Parallel Tempering
    cnt += PTF;
    for(int p = 0; p < int(cnt); p++){
      MEASURE.virtual_measure(CONF.Dx, CONF.Dy, CONF.Dz, Po, Ms, Es, HHs);
      tempering_simple(Ms, Es, accept1);
      for(int t = 0; t < Pnum; t++){
        HHs[t] = Hls[0][Po[t]];
        invTs[t] = 1.0/Tls[0][Po[t]];
      }
      for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
        cudaSetDevice(device_0 + gpu_i);
        CudaSafeCall(cudaMemcpyAsync(DinvTs[gpu_i], invTs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
        CudaSafeCall(cudaMemcpyAsync(DHs[gpu_i], HHs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
      }
    }
    if(int(cnt))
    cnt = 0;
  }

  //Do measurements (annealing)

  //int *accept = (int*)calloc(Pnum - 1, sizeof(int));
  int *accept = (int*)calloc((Tnum - 1)*Hnum + Tnum*(Hnum - 1), sizeof(int));
  for(C_i = 0 ; C_i < Cnum ; C_i ++){

    for(int t = 0; t < Pnum; t++){
      HHs[t] = Hls[C_i][Po[t]];
      invTs[t] = 1.0/Tls[C_i][Po[t]];
    }
    for (int i = 0; i< (Tnum - 1)*Hnum + Tnum*(Hnum - 1); i++) accept[i] = 0;
    for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
      cudaSetDevice(device_0 + gpu_i);
      CudaSafeCall(cudaMemcpyAsync(DinvTs[gpu_i], invTs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
      CudaSafeCall(cudaMemcpyAsync(DHs[gpu_i], HHs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
    }

    for(int i = 0; i < EQUI_Ni; i++){
      for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
        cudaSetDevice(device_0 + gpu_i);
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF1(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF2(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF3(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
      }
      cnt += PTF;
      for(int p = 0; p < int(cnt); p++){
        MEASURE.virtual_measure(CONF.Dx, CONF.Dy, CONF.Dz, Po, Ms, Es, HHs);
        tempering_simple(Ms, Es, accept1);
        for(int t = 0; t < Pnum; t++){
          HHs[t] = Hls[C_i][Po[t]];
          invTs[t] = 1.0/Tls[C_i][Po[t]];
        }
        for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
          cudaSetDevice(device_0 + gpu_i);
          CudaSafeCall(cudaMemcpyAsync(DinvTs[gpu_i], invTs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
          CudaSafeCall(cudaMemcpyAsync(DHs[gpu_i], HHs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
        }
      }
      if(int(cnt))
        cnt = 0;
    }
    cnt = 0;
    for(int b = 0; b < BIN_NUM; b++){
      //Take the ensemble average
      for(int i = 0; i < BIN_SZ; i++){
        for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
          cudaSetDevice(device_0 + gpu_i);
					for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
						cudaSetDevice(device_0 + gpu_i);
						SSF1(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
					}
					for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
						cudaSetDevice(device_0 + gpu_i);
						SSF2(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
					}
					for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
						cudaSetDevice(device_0 + gpu_i);
						SSF3(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
					}
        }
        MEASURE.measure(CONF.Dx, CONF.Dy, CONF.Dz, Po, Ms, Es, HHs);
/*
#ifdef GET_CORR
        if ( i % f_CORR==0){
          CORR.extract(Po, CONF);//==
        }
#endif
*/
        cnt += PTF;
        for(int p = 0; p < int(cnt); p++){
          tempering(Ms, Es, accept, staytmp, stay);
          for(int t = 0; t < Pnum; t++){
            HHs[t] = Hls[C_i][Po[t]];
            invTs[t] = 1.0/Tls[C_i][Po[t]];
            if (stay[Po[t]] > staylargest[Po[t]]){
              CONF.Dominatestateback(Po[t],t);
              staylargest[Po[t]] = stay[Po[t]];
            }
          }
          for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
            cudaSetDevice(device_0 + gpu_i);
            CudaSafeCall(cudaMemcpyAsync(DinvTs[gpu_i], invTs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
            CudaSafeCall(cudaMemcpyAsync(DHs[gpu_i], HHs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
          }
        }
        if(int(cnt))
          cnt = 0;
      }
      MEASURE.normalize_and_save_and_reset();
    }
    for (int iii = 0 ; iii < Pnum; iii ++){
      ivPo[Po[iii]] = iii;
    }
    //CONF.backtoHost(); //watch out! it must be compatible with the
    CONF.writedata();
#ifdef GET_CORR
    sprintf(Corrfn, "%s/Corr_%d", dir, b);
    CORR.changefile(Corrfn);
    for(int i = 0; i < CORR_N * f_CORR; i++){
      for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
        cudaSetDevice(device_0 + gpu_i);
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF1(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF2(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF3(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
      }
      if ( i % f_CORR==0){
        CORR.extract(Po, CONF);//==
      }
    }
    CORR.avg_write_reset(Po);
    for(int i = 0; i < 1000; i++){
      for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
        cudaSetDevice(device_0 + gpu_i);
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF1(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF2(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
				for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
					cudaSetDevice(device_0 + gpu_i);
					SSF3(CONF.Dx[gpu_i], CONF.Dy[gpu_i], CONF.Dz[gpu_i], seedDevice[gpu_i], DHs[gpu_i], DinvTs[gpu_i], stream[gpu_i]);
				}
      }
      //Parallel Tempering
      cnt += PTF;
      for(int p = 0; p < int(cnt); p++){
        MEASURE.virtual_measure(CONF.Dx, CONF.Dy, CONF.Dz, Po, Ms, Es, HHs);
        tempering_simple(Ms, Es, accept);
        for(int t = 0; t < Pnum; t++){
          HHs[t] = Hls[0][Po[t]];
          invTs[t] = 1.0/Tls[0][Po[t]];
        }
        for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
          cudaSetDevice(device_0 + gpu_i);
          CudaSafeCall(cudaMemcpyAsync(DinvTs[gpu_i], invTs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
          CudaSafeCall(cudaMemcpyAsync(DHs[gpu_i], HHs+gpu_i*Pnum_s, params_mem_size_s, cudaMemcpyHostToDevice, stream[gpu_i]));
        }
      }
      if(int(cnt))
      cnt = 0;
    }
#endif
  }
  char Histfn[128];
  sprintf(Histfn, "%s/%s", dir, "EHistogram");
  FILE *f_hist = fopen(Histfn, "w");
  fwrite(MEASURE.EHistogram, sizeof(unsigned int),Slice_NUM * Pnum, f_hist);
  fclose(f_hist);
  sprintf(Histfn, "%s/%s", dir, "ChernHistogram");
  FILE *f_chist = fopen(Histfn, "w");
  fwrite(MEASURE.ChernHistogram, sizeof(unsigned int), Slice_CNUM * Pnum, f_chist);
  fclose(f_chist);
  free(Ms);
  sdkStopTimer(&timer);
  double time = 1.0e-3 * sdkGetTimerValue(&timer);
    printf("G\n");
    fflush(stdout);


//======================= print details ==========================
  char detailFn[128];
  sprintf(detailFn, "%s/details", dir);
  FILE *detailFp = fopen(detailFn, "w");
  fprintf(detailFp, "elapsed time = %f (sec)\n", time);
  double speed = 0;
  speed = (H_N / time / 1000000000) * ((EQUI_Ni + BIN_SZ * BIN_NUM) * Cnum + EQUI_N) * Pnum;
  fprintf(detailFp, "speed = %f (GHz)\n", speed);
  fprintf(detailFp, "number of gpu = %d\n", StreamN);
  fprintf(detailFp, "RNG: WarpStandard\n", H_SpinSize);
  fprintf(detailFp, "SpinSize = %d\n", H_SpinSize);
  fprintf(detailFp, "A = %4.3f\n", H_A);
  fprintf(detailFp, "D_Rashba = %4.3f\n", DR);
  fprintf(detailFp, "D_Dresselhaus = %4.3f\n", DD);
  fprintf(detailFp, "BlockSize_x = %d\n", H_BlockSize_x);
  fprintf(detailFp, "BlockSize_y = %d\n", H_BlockSize_y);
  fprintf(detailFp, "GridSize_x = %d\n", H_GridSize_x);
  fprintf(detailFp, "GridSize_y = %d\n", H_GridSize_y);
  fprintf(detailFp, "Bin Size = %d\n", BIN_SZ);
  fprintf(detailFp, "Bin Number = %d\n", BIN_NUM);
  fprintf(detailFp, "Equilibration N = %d\n", EQUI_N);
  fprintf(detailFp, "Equilibration Ni = %d\n", EQUI_Ni);
  fprintf(detailFp, "f_CORR = %d\n", f_CORR);
  fprintf(detailFp, "CORR_N = %d\n", CORR_N);
  fprintf(detailFp, "PT frequency = %3.2f\n", PTF);
  fprintf(detailFp, "Pnum = %d\n", Pnum);
  fprintf(detailFp, "Temperature Set: ");
  for(int i = 0; i < Cnum; i++){
    for(int j = 0; j < Pnum; j++){
      fprintf(detailFp, "%.5f  ", Tls[i][j]);
    }
  }
  fprintf(detailFp, "\n");
  fprintf(detailFp, "field Set: ");
  for(int i = 0; i < Cnum; i++){
    for(int j = 0; j < Pnum; j++){
      fprintf(detailFp, "%.5f  ", Hls[i][j]/(DR*DR + DD*DD));
    }
  }
  for(int i = 0; i < Pnum; i++){
    fprintf(detailFp, "\n");
    fprintf(detailFp, "Po[%d]=%d",i,Po[i]);
  }
  fprintf(detailFp, "\n");
  fprintf(detailFp, "Acceptance rates: ");
  if (PTF != 0 ){
    for(int i = 0; i < (Tnum - 1)*Hnum + Tnum*(Hnum - 1); i++)
      fprintf(detailFp, "%4.3f  ", float(accept[i]) / (BIN_SZ * BIN_NUM * PTF));
  }
  fprintf(detailFp, "\n");
  if (ORDER){
    fprintf(detailFp, "Configurations start from ordered state.\n");
  }
  else {
    fprintf(detailFp, "Configurations start from random state.\n");
  }
  fprintf(detailFp, "N_histE = %d\n", Slice_NUM);
  fprintf(detailFp, "E_lowest = %4.3f\n", E_lowest);
  fprintf(detailFp, "E_highest = %4.3f\n", E_highest);
  fprintf(detailFp, "N_histChern = %d\n", Slice_CNUM);
  fprintf(detailFp, "Chern_lowest = %4.3f\n", Chern_lowest);
  fprintf(detailFp, "Chern_highest = %4.3f\n", Chern_highest);
  fprintf(detailFp, "Done by Po-Kuan Wu ^_^\n", EQUI_N);
  fclose(detailFp);
//===================== print details end =========================

  //Set free memory
  free(seedHost);
  for (gpu_i = 0; gpu_i < StreamN; gpu_i++){
    cudaSetDevice(device_0 + gpu_i);
    CudaSafeCall(cudaFree(DinvTs[gpu_i]));
    CudaSafeCall(cudaFree(DHs[gpu_i]));
    CudaSafeCall(cudaFree(seedDevice[gpu_i]));
  }
  //CORR.~correlation();
  //MEASURE.~measurements();
  //CONF.~configuration();
  return 0;
}


//=============================== functions ==================================
void tempering_simple(double *Ms, double *Es, int *accept){
  int map[Pnum];	//map[t] the configuration of t'th temperature
  int i, j, tmp, partT_num = (Tnum - 1) * Hnum;
  double tmpEM;

  for(i = 0; i < Pnum; i++)
    map[Po[i]] = i;

  double delta;
  int flag = 0;
  for(i = 0; i < Tnum; i++){
    for (j = 0; j < Hnum; j++){
      //T excnange
      if (i < Tnum -1){
	delta = (Es[j * Tnum + i] - Es[j * Tnum + i + 1]) * ((1.0 / Tls[C_i][j*Tnum + i]) - (1.0 / Tls[C_i][j*Tnum +i + 1]));
	if(delta > 0)
	  flag = 1;
	else if(uni01_sampler() < exp(delta))
	  flag = 1;
	if(flag){
	  tmp = Po[map[j * Tnum + i]];
	  Po[map[j * Tnum + i]] = Po[map[j * Tnum + i + 1]];
	  Po[map[j * Tnum + i + 1]] = tmp;
	  tmp = map[j * Tnum + i];
	  map[j * Tnum + i] = map[j * Tnum + i + 1];
	  map[j * Tnum + i + 1] = tmp;
	  tmpEM = Es[j * Tnum + i];
	  Es[j * Tnum + i] = Es[j * Tnum + i + 1];
	  Es[j * Tnum + i + 1] = tmpEM;
	  tmpEM = Ms[j * Tnum + i];
	  Ms[j * Tnum + i] = Ms[j * Tnum + i + 1];
	  Ms[j * Tnum + i + 1] = tmpEM;
	  accept[j * (Tnum - 1) + i] += 1;
	  flag = 0;
	}
      }
    }
  }
  for(i = 0; i < Tnum; i++){
    for (j = 0; j < Hnum; j++){
      //H excnange
      if (j < Hnum -1){
        delta = (Ms[(j + 1) * Tnum + i] - Ms[j * Tnum + i]) * ( Hls[C_i][j * Tnum + i] - Hls[C_i][(j + 1) * Tnum + i]) / Tls[C_i][j * Tnum + i];
        if(delta > 0)
          flag = 1;
        else if(uni01_sampler() < exp(delta))
          flag = 1;
        if(flag){
          tmp = Po[map[j * Tnum + i]];
          Po[map[j * Tnum + i]] = Po[map[(j + 1) * Tnum + i]];
          Po[map[(j + 1) * Tnum + i]] = tmp;
          tmp = map[j * Tnum + i];
          map[j * Tnum + i] = map[(j + 1) * Tnum + i];
          map[(j + 1) * Tnum + i] = tmp;
          tmpEM = Es[j * Tnum + i];
          Es[j * Tnum + i] = Es[(j + 1) * Tnum + i];
          Es[(j + 1) * Tnum + i] = tmpEM;
          tmpEM = Ms[j * Tnum + i];
          Ms[j * Tnum + i] = Ms[(j + 1) * Tnum + i];
          Ms[(j + 1) * Tnum + i] = tmpEM;
          accept[partT_num + j * Tnum + i] += 1;
          flag = 0;
        }
      }
    }
  }
}

//=============================== functions ==================================
void tempering(double *Ms, double *Es, int *accept, int *staytmp, int *stay){
  int map[Pnum];	//map[t] the configuration of t'th temperature
  int i, j, tmp, partT_num = (Tnum - 1) * Hnum;
  double tmpEM;

  for(i = 0; i < Pnum; i++)
    map[Po[i]] = i;

  double delta;
  int flag = 0;
  for(i = 0; i < Tnum; i++){
    for (j = 0; j < Hnum; j++){
	staytmp[j * Tnum + i] = 1;
    }
  }
  for(i = 0; i < Tnum; i++){
    for (j = 0; j < Hnum; j++){
      //T excnange
      if (i < Tnum -1){
	delta = (Es[j * Tnum + i] - Es[j * Tnum + i + 1]) * ((1.0 / Tls[C_i][j*Tnum + i]) - (1.0 / Tls[C_i][j*Tnum +i + 1]));
	if(delta > 0)
	  flag = 1;
	else if(uni01_sampler() < exp(delta))
	  flag = 1;
	if(flag){
	  tmp = Po[map[j * Tnum + i]];
	  Po[map[j * Tnum + i]] = Po[map[j * Tnum + i + 1]];
	  Po[map[j * Tnum + i + 1]] = tmp;
	  tmp = map[j * Tnum + i];
	  map[j * Tnum + i] = map[j * Tnum + i + 1];
	  map[j * Tnum + i + 1] = tmp;
	  tmpEM = Es[j * Tnum + i];
	  Es[j * Tnum + i] = Es[j * Tnum + i + 1];
	  Es[j * Tnum + i + 1] = tmpEM;
	  tmpEM = Ms[j * Tnum + i];
	  Ms[j * Tnum + i] = Ms[j * Tnum + i + 1];
	  Ms[j * Tnum + i + 1] = tmpEM;
	  accept[j * (Tnum - 1) + i] += 1;
	  flag = 0;
	  staytmp[j * Tnum + i] *= 0;
	  staytmp[j * Tnum + i + 1] *= 0;
	}
      }
    }
  }
  for(i = 0; i < Tnum; i++){
    for (j = 0; j < Hnum; j++){
      //H excnange
      if (j < Hnum -1){
        delta = (Ms[(j + 1) * Tnum + i] - Ms[j * Tnum + i]) * ( Hls[C_i][j * Tnum + i] - Hls[C_i][(j + 1) * Tnum + i]) / Tls[C_i][j * Tnum + i];
        if(delta > 0)
          flag = 1;
        else if(uni01_sampler() < exp(delta))
          flag = 1;
        if(flag){
          tmp = Po[map[j * Tnum + i]];
          Po[map[j * Tnum + i]] = Po[map[(j + 1) * Tnum + i]];
          Po[map[(j + 1) * Tnum + i]] = tmp;
          tmp = map[j * Tnum + i];
          map[j * Tnum + i] = map[(j + 1) * Tnum + i];
          map[(j + 1) * Tnum + i] = tmp;
          tmpEM = Es[j * Tnum + i];
          Es[j * Tnum + i] = Es[(j + 1) * Tnum + i];
          Es[(j + 1) * Tnum + i] = tmpEM;
          tmpEM = Ms[j * Tnum + i];
          Ms[j * Tnum + i] = Ms[(j + 1) * Tnum + i];
          Ms[(j + 1) * Tnum + i] = tmpEM;
          accept[partT_num + j * Tnum + i] += 1;
          flag = 0;
	  staytmp[j * Tnum + i] *= 0;
	  staytmp[(j + 1) * Tnum + i] *= 0;
        }
      }
    }
  }
  for(i = 0; i < Tnum; i++){
    for (j = 0; j < Hnum; j++){
      stay[j * Tnum + i] = staytmp[j * Tnum + i]?(stay[j * Tnum + i]+1):0;
    }
  }
}
void var_examine(){
#ifndef TRI
  if(H_SpinSize % (H_BlockSize_x * 2) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", H_BlockSize_x * 2);
    exit(0);
  }
  if(H_SpinSize % (H_BlockSize_y * 2) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", H_BlockSize_y * 2);
    exit(0);
  }
#endif
#ifdef TRI
  if(H_SpinSize % (H_BlockSize_x * 3) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", H_BlockSize_x * 2);
    exit(0);
  }
  if(H_SpinSize % (H_BlockSize_y * 3) != 0){
    fprintf(stderr, "SpinSize must be the multiple of %d\n", H_BlockSize_y * 2);
    exit(0);
  }
#endif
#ifndef THIN
  if (H_SpinSize_z != 1){
    fprintf(stderr, "SpinSize_z must be 1 %d\n", H_BlockSize_y * 2);
    exit(0);
  }
#endif
}
