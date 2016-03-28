#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <fcntl.h>
#include <stdint.h>
#include <time.h>
#include <helper_timer.h>
#include <math.h>
#include <cuda.h>
#include <vector>
#include "WarpStandard.cuh"
using namespace std;
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
using namespace boost;
unsigned seed = 73;
mt19937 rng(seed);
uniform_01<mt19937> uni01_sampler(rng);
//--------variables for one temperature replica----------
#define SpinSize 32                          //Each thread controls 2 by 2 by 2 spins
#define SpinSize_z 8
#define BlockSize_x 16
#define BlockSize_y 16
#define GridSize_x (SpinSize/BlockSize_x/2)
#define GridSize_y (SpinSize/BlockSize_y/2)
#define N (SpinSize*SpinSize*(SpinSize_z))              //The number of spins of the system + boundary effective spins
#define Nplane (SpinSize*SpinSize)              //The number of spins of the system
#define TN (Nplane / 4)									//The number of needed threads
#define BN (GridSize_x*GridSize_y)       //The number of needed blocks
//---------------------End-------------------------------
#define BIN_SZ 10000//
#define BIN_NUM 20
#define EQUI_N 100000////16000000
#define relax_N 0
#define CORR_L 0
#define MEASURE_NUM 5
#define coo(k, j, i) ((k) * Nplane + (j) * SpinSize + (i))

/*
#define Jxx (-0.8)
#define Jyy (-0.8)
#define Jzz (-1.0)
#define Jxy (0.0)
#define Jyx (0.0)
#define Jyz (0.0)
#define Jzy (0.0)
#define Jxz (0.0)
#define Jzx (0.0)
#define H (6.0)
*/
#define NORM (float(4.656612873077393e-10)) // UINT_MAX * NORM = 2
#define TOPI (float(1.462918078360668e-9))
#define TWOPI (float(6.28318530717956))	//2*pi
/*
#define BXPxx(BD) Jxx
#define BYPxx(BD) Jxx
#define BXMxx(BD) Jxx
#define BYMxx(BD) Jxx
#define BXPyy(BD) Jyy
#define BYPyy(BD) Jyy
#define BXMyy(BD) Jyy
#define BYMyy(BD) Jyy
#define BXPzz(BD) Jzz
#define BYPzz(BD) Jzz
#define BXMzz(BD) Jzz
#define BYMzz(BD) Jzz
#define BXPxy(BD) Jxy
#define BYPxy(BD) Jxy
#define BXMxy(BD) Jxy
#define BYMxy(BD) Jxy
#define BXPyx(BD) Jyx
#define BYPyx(BD) Jyx
#define BXMyx(BD) Jyx
#define BYMyx(BD) Jyx
#define BXPyz(BD) Jyz
#define BYPyz(BD) Jyz
#define BXMyz(BD) Jyz
#define BYMyz(BD) Jyz
#define BXPzy(BD) Jzy
#define BYPzy(BD) Jzy
#define BXMzy(BD) Jzy
#define BYMzy(BD) Jzy
#define BXPzx(BD) Jzx
#define BYPzx(BD) Jzx
#define BXMzx(BD) Jzx
#define BYMzx(BD) Jzx
#define BXPxz(BD) Jxz
#define BYPxz(BD) Jxz
#define BXMxz(BD) Jxz
#define BYMxz(BD) Jxz
*/
#define Hfinal (0.016000)
#define A (0.171573)//(-DR * DR)
#define DR (0.0)//(1.41421)//(0.585786)//(0.8)////(0.4)//(1.02749)
#define DD (0.585786)//(1.41421)//(0.8)////(0.4)//(1.02749)
#define BXPxx (1.000000)
#define BYPxx (1.000000)
#define BZPxx (1.000000)
#define BXMxx (1.000000)
#define BYMxx (1.000000)
#define BZMxx (1.000000)
#define BXPyy (1.000000)
#define BYPyy (1.000000)
#define BZPyy (1.000000)
#define BXMyy (1.000000)
#define BYMyy (1.000000)
#define BZMyy (1.000000)
#define BXPzz (1.000000)
#define BYPzz (1.000000)
#define BZPzz (1.000000)
#define BXMzz (1.000000)
#define BYMzz (1.000000)
#define BZMzz (1.000000)

#define BXPxy (0.000000)
#define BYPxy (0.000000)
#define BXMxy (0.000000)
#define BYMxy (0.000000)
#define BXPyx (0.000000)
#define BYPyx (0.000000)
#define BXMyx (0.000000)
#define BYMyx (0.000000)

#define BZPyx (-DD)
#define BZMyx (DD)
#define BZPxy (DD)
#define BZMxy (-DD)

#define BXPyz (DD)
#define BYPyz (DR)
#define BXMyz (-DD)
#define BYMyz (-DR)
#define BXPzy (-DD)
#define BYPzy (-DR)
#define BXMzy (DD)
#define BYMzy (DR)

#define BXPzx (-DR)
#define BYPzx (DD)
#define BXMzx (DR)
#define BYMzx (-DD)
#define BXPxz (DR)
#define BYPxz (-DD)
#define BXMxz (-DR)
#define BYMxz (DD)
#define ID "skyr_d16z8AO_annealingT_thin"
#define PTF	(float(0.00))	//Frequency of parallel tempering

#include "flip1.cu"
#include "flip2.cu"
//#include "relax1.cu"
//#include "relax2.cu"
#include "cals.cu"
#include "corr.cu"
#include "extend.cu"
#define EQUI_Ni (400000)//)
#define GET_CORR
#define f_CORR (500)

void tempering(double*, int*);
unsigned int block = BlockSize_x * BlockSize_y;
unsigned int grid;
vector<float> Tls;
vector<float> Hls;
vector<int>Ho;		//order of Temperature, Tls[To[t]] is the temperature of t'th configuration.
vector<int>ivHo;		//order of Temperature, Tls[To[t]] is the temperature of t'th configuration.
unsigned int Tnum;
unsigned int Hnum;
float Tcurrent = 0.8;

int main(int argc, char *argv[]){
	//initialize
  if (setDev()==1){
    return 1;

  }
	if(SpinSize % (BlockSize_x * 2) != 0){
		fprintf(stderr, "SpinSize must be the multiple of %d\n", BlockSize_x * 2);
		exit(0);
	}
	if(SpinSize % (BlockSize_y * 2) != 0){
		fprintf(stderr, "SpinSize must be the multiple of %d\n", BlockSize_y * 2);
		exit(0);
	}
	if(CORR_L > SpinSize / 2){
		fprintf(stderr, "The length for correlation measurement must be smaller than half the system size.\n");
		exit(0);
	}
	//read in temperatures
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
		i = 0;
		Temp_mem_size = Tnum * sizeof(float);
		Tfp = fopen(argv[2], "r");
		while(fscanf(Tfp, "%f", &tmp) != EOF){
			Hls.push_back((DD * DD + DR * DR)*tmp);
			Ho.push_back(i);
			ivHo.push_back(i);
			i++;
		}
		fclose(Tfp);
		Hnum = Hls.size();
		H_mem_size = Hnum * sizeof(float);
		grid = Hnum * BN;
	}
	else{
		fprintf(stderr, "Give me a temperature set!!!\n");
		fprintf(stderr, "Give me a field set!!!\n");
		exit(0);
	}
	//invTs is the inverse temperature in order of configurations on GPU.
	float *HHs;
	float *DHs;
	int *DTo;
	cudaMalloc((void**)&DTo, Hnum * sizeof(int));
	HHs = (float*)malloc(H_mem_size);
	cudaMalloc((void**)&DHs, H_mem_size);
	//Declare sizes
	unsigned int Spin_mem_size = Hnum * N * sizeof(float);
	//unsigned int Corr_mem_size = Tnum * CORR_L * CORR_L * sizeof(double);
	unsigned int Out_mem_size = Hnum * MEASURE_NUM * BN * sizeof(double);
	unsigned int totalRngs = Hnum * TN / WarpStandard_K;
	float *Hconfx = (float*)malloc(Spin_mem_size);
	float *Hconfy = (float*)malloc(Spin_mem_size);
	float *Hconfz = (float*)malloc(Spin_mem_size);
	double *Hout = (double*)malloc(Out_mem_size);
#ifdef GET_CORR
	unsigned int Spin_mem_size_p = Hnum * Nplane * sizeof(float);
	unsigned int Spin_mem_size_d = Hnum * Nplane * sizeof(double);
	int corrcount = 0;
	double *HSum_corr = (double*)malloc(Spin_mem_size_d);
#endif
	//double *Hcorr = (double*)malloc(Corr_mem_size);
	unsigned seedBytes = totalRngs * sizeof(unsigned int) * WarpStandard_STATE_WORDS;
	unsigned int *seedDevice = 0;
	if(cudaMalloc((void **)&seedDevice, seedBytes)){
		fprintf(stderr, "Error couldn't allocate state array of size %u\n", seedBytes);
		exit(1);
	}
	unsigned int* seedHost = (unsigned int*)malloc(seedBytes);
	srand(seed);
	for(int i = 0; i < seedBytes / sizeof(unsigned int); i++)
		seedHost[i] = uni01_sampler() * UINT_MAX;
	cudaMemcpy(seedDevice, seedHost, seedBytes, cudaMemcpyHostToDevice);
	//Allocate device memory
	float *Dconfx;
	float *Dconfy;
	float *Dconfz;
	double *Dout;
	//double *Dcorr;
	if(cudaMalloc((void**)&Dconfx, Spin_mem_size)){
		fprintf(stderr, "Error couldn't allocate Device Memory!!\n");
		exit(1);
	}
	if(cudaMalloc((void**)&Dconfy, Spin_mem_size)){
		fprintf(stderr, "Error couldn't allocate Device Memory!!\n");
		exit(1);
	}
	if(cudaMalloc((void**)&Dconfz, Spin_mem_size)){
		fprintf(stderr, "Error couldn't allocate Device Memory!!\n");
		exit(1);
	}
	cudaMalloc((void**)&Dout, Out_mem_size);
	//cudaMalloc((void**)&Dcorr, Corr_mem_size);
#ifdef GET_CORR
	float *Dcorr;
	double *DSum_corr;
	if(cudaMalloc((void**)&Dcorr, Spin_mem_size_p)){
		fprintf(stderr, "Error couldn't allocate Device Memory!!\n");
		exit(1);
	}

	if(cudaMalloc((void**)&DSum_corr, Spin_mem_size_d)){
		fprintf(stderr, "Error couldn't allocate Device Memory!!\n");
		exit(1);
	}
#endif

	//Set up output data path
	char dir[128];
	char conf_dir[128];
	sprintf(dir, "Data/L_%d-%s", SpinSize, ID);
	sprintf(conf_dir, "Conf/L_%d-%s", SpinSize, ID);
	mkdir(dir, 0755);
	mkdir(conf_dir, 0755);
	char Seedfn[128];
	sprintf(Seedfn, "Conf/L_%d-%s/seed", SpinSize, ID);
	int seedfd = open(Seedfn, O_CREAT | O_WRONLY, 0644);
	write(seedfd, seedHost, seedBytes);
	close(seedfd);
	char Efn[128];
	char Mfn[128];
	char Chernfn[128];
	char E2fn[128];
	char E4fn[128];
	char M2fn[128];
	char M4fn[128];
	//char Helfn[128];
	//char Corrfn[128];
	char Confxfn[128];
	char Confyfn[128];
	char Confzfn[128];
	sprintf(Efn, "%s/E", dir);
	sprintf(Mfn, "%s/M", dir);
	sprintf(Chernfn, "%s/Chern", dir);
	sprintf(E2fn, "%s/E2", dir);
	sprintf(E4fn, "%s/E4", dir);
	sprintf(M2fn, "%s/M2", dir);
	sprintf(M4fn, "%s/M4", dir);
	//sprintf(Helfn, "%s/Hel", dir);
	sprintf(Confxfn, "%s/Confx", conf_dir);
	sprintf(Confyfn, "%s/Confy", conf_dir);
	sprintf(Confzfn, "%s/Confz", conf_dir);
#ifdef GET_CORR
  char Corrfn[128];
	sprintf(Corrfn, "%s/Corr", dir);
	int Corrfd = open(Corrfn, O_CREAT | O_WRONLY, 0644);
#endif

	StopWatchInterface *timer=NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	//Give initial configuration and settle the systems down to equilibrium states
	double pi = 3.141592653589793;
	double th, phi;
	for(int i = 0; i < N * Hnum; i++){
		th = uni01_sampler() * pi;
		phi = uni01_sampler() * 2 * pi;
		Hconfx[i] = cos(th);
		th = sin(th);
		Hconfy[i] = th * cos(phi);
		Hconfz[i] = th * sin(phi);
    /*
		Hconfx[i] = 0;
		Hconfy[i] = 0;
		Hconfz[i] = 1;
    */
	}
	for(int i = 0; i < Nplane * Hnum; i++){
#ifdef GET_CORR
    HSum_corr[i] = 0.0; //initialize
#endif
	}
	cudaMemcpy(Dconfx, Hconfx, Spin_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(Dconfy, Hconfy, Spin_mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(Dconfz, Hconfz, Spin_mem_size, cudaMemcpyHostToDevice);
#ifdef GET_CORR
	cudaMemcpy(DSum_corr, HSum_corr, Spin_mem_size_d, cudaMemcpyHostToDevice);
#endif
  int Eqii = 0;//150;
	for(int i = 0; i < Hnum; i++)
		HHs[i] = Hls[i];
	cudaMemcpy(DHs, HHs, H_mem_size, cudaMemcpyHostToDevice);
  double *Ms = (double*)malloc(Hnum * sizeof(double));
  double E;
	int *accept1 = (int*)calloc(Hnum - 1, sizeof(int));
	float cnt = 0;
  Tcurrent = Tls[Tls.size()-1];

	for(int i = 0; i < EQUI_N; i++){
		flipTLBR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, DHs, 1.0 / Tcurrent);
		flipBLTR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, DHs, 1.0 / Tcurrent);
		/*
		for (int q = 0; q < relax_N; q++){
			relaxTLBR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, H);
			relaxBLTR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, H);
		}*/
		//================================= no PT ==========================================
		/*
    cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout); //, Dcorr);
    cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost);
    for(int t = 0; t < Hnum; t++){
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
      for(int t = 0; t < Hnum; t++) HHs[t] = Hls[Ho[t]];
      cudaMemcpy(DHs, HHs, H_mem_size, cudaMemcpyHostToDevice);
    }
    if(int(cnt))
      cnt = 0;
    */
	}

	//Do measurements (annealing)
	unsigned int data_mem_size = Hnum * sizeof(double);
	int Efd = open(Efn, O_CREAT | O_WRONLY, 0644);
	int Mfd = open(Mfn, O_CREAT | O_WRONLY, 0644);
	int Chernfd = open(Chernfn, O_CREAT | O_WRONLY, 0644);
	int E2fd = open(E2fn, O_CREAT | O_WRONLY, 0644);
	int E4fd = open(E4fn, O_CREAT | O_WRONLY, 0644);
	int M2fd = open(M2fn, O_CREAT | O_WRONLY, 0644);
	int M4fd = open(M4fn, O_CREAT | O_WRONLY, 0644);
	//int Helfd = open(Helfn, O_CREAT | O_WRONLY, 0644);
	double *Eout = (double*)malloc(data_mem_size);
	double *Mout = (double*)malloc(data_mem_size);
	double *Chernout = (double*)malloc(data_mem_size);
	double *E2out = (double*)malloc(data_mem_size);
	double *E4out = (double*)malloc(data_mem_size);
	double *M2out = (double*)malloc(data_mem_size);
	double *M4out = (double*)malloc(data_mem_size);
	int Confxfd = open(Confxfn, O_CREAT | O_WRONLY, 0644);
	int Confyfd = open(Confyfn, O_CREAT | O_WRONLY, 0644);
	int Confzfd = open(Confzfn, O_CREAT | O_WRONLY, 0644);
	int *accept = (int*)calloc(Hnum - 1, sizeof(int));
	//double *Helout = (double*)malloc(data_mem_size);
	for(int T_i = 0 ; T_i < Tnum ; T_i ++){
		for (int i = 0; i< Hnum-1; i++) accept[i] = 0;
		Tcurrent = Tls [Tls.size() - 1 - T_i];
    for(int i = 0; i < EQUI_Ni; i++){
      flipTLBR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, DHs, 1.0 / Tcurrent);
      flipBLTR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, DHs, 1.0 / Tcurrent);
      /*
      for (int q = 0; q < relax_N; q++){
        relaxTLBR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, H);
        relaxBLTR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, H);
      }*/
      //======================= no PT ===============================
      /*
      cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout); //, Dcorr);
      cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost);
      for(int t = 0; t < Hnum; t++){
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
        for(int t = 0; t < Hnum; t++) HHs[t] = Hls[Ho[t]];
        cudaMemcpy(DHs, HHs, H_mem_size, cudaMemcpyHostToDevice);
      }
      if(int(cnt))
        cnt = 0;
      */
    }
    cnt = 0;
    for(int b = 0; b < BIN_NUM; b++){
      //printf("b = %d\n", b);
      //Take the ensemble average
      double E2;
      double Mx, My, Mz, M2, Chern;
      //double Helx, Hely, Helz;
      double *binE = (double*)calloc(Hnum, sizeof(double));
      double *binM = (double*)calloc(Hnum, sizeof(double));
      double *binChern = (double*)calloc(Hnum, sizeof(double));
      double *binE2 = (double*)calloc(Hnum, sizeof(double));
      double *binM2 = (double*)calloc(Hnum, sizeof(double));
      double *binE4 = (double*)calloc(Hnum, sizeof(double));
      double *binM4 = (double*)calloc(Hnum, sizeof(double));
      //double *binHel = (double*)calloc(Tnum, sizeof(double));
      //cudaMemset((void*)Dcorr, 0, Corr_mem_size);
      for(int i = 0; i < BIN_SZ; i++){
        flipTLBR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, DHs, 1.0 / Tcurrent);
        flipBLTR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, DHs, 1.0 / Tcurrent);
        /*
        for (int q = 0; q < relax_N; q++){
          relaxTLBR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, H);
          relaxBLTR<<<grid, block>>>(Dconfx, Dconfy, Dconfz, seedDevice, H);
        }*/
        cal<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dout); //, Dcorr);
#ifdef GET_CORR
        if ( i % f_CORR==0){
          cudaMemcpy(DTo, &Ho[0], Tnum * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemset(Dcorr, 0, Spin_mem_size);
          for (int labelx = 0; labelx < SpinSize; labelx += 4){
            for (int labely = 0; labely < SpinSize; labely += 4){
              getcorr<<<grid, block>>>(Dconfx, Dconfy, Dconfz, Dcorr, labelx, labely);
            }
          }
          sumcorr<<<grid, block>>>(DSum_corr, Dcorr, DTo);
          corrcount += 1;
        }
#endif
        cudaMemcpy(Hout, Dout, Out_mem_size, cudaMemcpyDeviceToHost);
        for(int t = 0; t < Hnum; t++){
          int raw_off = t * MEASURE_NUM * BN;
          E = 0;
          //Helx = 0, Hely = 0, Helz = 0;
          Mx = 0, My = 0, Mz = 0, Chern = 0;
          for(int j = 0; j < BN; j++)
            E += Hout[raw_off + j];
          //for(int j = BN; j < 2 * BN; j++)
          //	Helx += Hout[raw_off + j];
          //for(int j = 2 * BN; j < 3 * BN; j++)
          //	Hely += Hout[raw_off + j];
          //for(int j = 3 * BN; j < 4 * BN; j++)
          //	Helz += Hout[raw_off + j];
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
          binE[Ho[t]] += E;
          M2 = Mx * Mx + My * My + Mz * Mz;
          E2 = E * E;
          binM[Ho[t]] += sqrt(M2);
          binChern[Ho[t]] += Chern;
          binE2[Ho[t]] += E2;
          binM2[Ho[t]] += M2;
          binE4[Ho[t]] += E2 * E2;
          binM4[Ho[t]] += M2 * M2;
          //binHel[Ho[t]] += (Helx * Helx + Hely * Hely + Helz * Helz) / 3;
        }
        //Parallel Tempering
        /*
        cnt += PTF;
        for(int p = 0; p < int(cnt); p++){
          tempering(Ms, accept);
          for(int t = 0; t < Hnum; t++) HHs[t] = Hls[Ho[t]];
          cudaMemcpy(DHs, HHs, H_mem_size, cudaMemcpyHostToDevice);
        }
        if(int(cnt))
          cnt = 0;
        */
      }
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
      //write(Helfd, Helout, data_mem_size);
      //cudaMemcpy(Hcorr, Dcorr, Corr_mem_size, cudaMemcpyDeviceToHost);
      //write(corrfd, Hcorr, Corr_mem_size);
      //close(corrfd);
      free(binE);
      free(binM);
      free(binChern);
      free(binE2);
      free(binE4);
      free(binM2);
      free(binM4);
      //free(binHel);
    }
    cudaMemcpy(Hconfx, Dconfx, Spin_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Hconfy, Dconfy, Spin_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Hconfz, Dconfz, Spin_mem_size, cudaMemcpyDeviceToHost);
    for (int iii = 0 ; iii < Hnum; iii ++){
    	ivHo[Ho[iii]] = Ho[iii];
    }
    write(Confxfd, Hconfx, Spin_mem_size);
    write(Confyfd, Hconfy, Spin_mem_size);
    write(Confzfd, Hconfz, Spin_mem_size);
#ifdef GET_CORR
    avgcorr<<<grid, block>>>(DSum_corr, double(corrcount));
    cudaMemcpy(HSum_corr, DSum_corr, Spin_mem_size_d, cudaMemcpyDeviceToHost);
    write(Corrfd, HSum_corr, Spin_mem_size_d);
    cudaMemset((void*)DSum_corr, 0, Spin_mem_size_d);
#endif
  }
  free(Ms);
	sdkStopTimer(&timer);
    double time = 1.0e-3 * sdkGetTimerValue(&timer);
#ifdef GET_CORR
	close(Corrfd);
#endif
	close(Efd);
	close(Mfd);
	close(Chernfd);
	close(E2fd);
	close(E4fd);
	close(M2fd);
	close(M4fd);
	//close(Helfd);
	close(Confxfd);
	close(Confyfd);
	close(Confzfd);
	//close(corrfd);

	char detailFn[128];
	sprintf(detailFn, "%s/details", dir);
	FILE *detailFp = fopen(detailFn, "w");
	fprintf(detailFp, "elapsed time = %f (sec)\n", time);
	double speed = 0;
	speed = (N / time / 1000000000) * (BIN_SZ * BIN_NUM + EQUI_N) * Tnum;
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
	//fprintf(detailFp, "CORR_L = %d\n", CORR_L);
	fprintf(detailFp, "Bin Size = %d\n", BIN_SZ);
	fprintf(detailFp, "Bin Number = %d\n", BIN_NUM);
	fprintf(detailFp, "Equilibration N = %d\n", EQUI_N);
	fprintf(detailFp, "Equilibration Ni = %d\n", EQUI_Ni);
	fprintf(detailFp, "f_CORR = %d\n", f_CORR);
	fprintf(detailFp, "PT frequency = %3.2f\n", PTF);
	fprintf(detailFp, "Relaxation N = %d\n", relax_N);
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
	for(int i = 0; i < Hnum; i++){
    fprintf(detailFp, "\n");
    fprintf(detailFp, "Ho[%d]=%d",i,Ho[i]);
  }
	fprintf(detailFp, "\n");
	fprintf(detailFp, "Acceptance rates: ");
	if (PTF != 0 ){
	for(int i = 0; i < Tnum - 1; i++)
		fprintf(detailFp, "%4.3f  ", float(accept[i]) / (BIN_SZ * BIN_NUM * PTF));
	}
	fprintf(detailFp, "\n");
	fprintf(detailFp, "At all temperatures, configuration starts from ground state.\n");
	fprintf(detailFp, "Done by Po-Kuan Wu ^_^\n", EQUI_N);
	fclose(detailFp);

	//Set free memory
	free(Hconfx);
	free(Hconfy);
	free(Hconfz);
	free(Hout);
	//free(Hcorr);
	free(seedHost);
	free(Eout);
	free(Mout);
	free(Chernout);
	free(E2out);
	free(E4out);
	free(M2out);
	free(M4out);
	//free(Helout);
	cudaFree(DHs);
	cudaFree(Dconfx);
	cudaFree(Dconfy);
	cudaFree(Dconfz);
	cudaFree(Dout);
	//cudaFree(Dcorr);
	cudaFree(seedDevice);
#ifdef GET_CORR
	free(HSum_corr);
  cudaFree(Dcorr);
  cudaFree(DTo);
  cudaFree(DSum_corr);
#endif
	return 0;
}

void tempering(double *Ms, int *accept){
	int map[Hls.size()];	//map[t] the configuration of t'th temperature
	for(int i = 0; i < Hnum; i++){
		map[Ho[i]] = i;
  }
	double delta;
	int flag = 0;
	for(int i = 0; i < Hnum - 1; i++){
		delta = (Ms[i + 1] - Ms[i]) * ( Hls[i] - Hls[i + 1]) / Tcurrent;
		if(delta > 0)
			flag = 1;
		else if(uni01_sampler() < exp(delta))
			flag = 1;
		if(flag){
			int tmp = Ho[map[i]];
			Ho[map[i]] = Ho[map[i + 1]];
			Ho[map[i + 1]] = tmp;
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
