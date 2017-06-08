#include "params.cuh"
unsigned int block;
unsigned int grid;
unsigned int rngShmemsize;
unsigned int caloutputsize;
unsigned int H_SpinSize;
unsigned int H_SpinSize_z;
unsigned int H_BlockSize_x;
unsigned int H_BlockSize_y;
unsigned int H_GridSize_x;
unsigned int H_GridSize_y;
unsigned int H_N;
unsigned int H_Nplane;
unsigned int H_TN;
unsigned int H_BN;
//------ system size setting end --------

//------ gpu setting -------
int StreamN;
int device_0;
cudaStream_t stream[10];
//---- gpu setting end -----

//------ system variable setting --------
//!!!!!!!!!!!!notice that the value of DD and DR are set while compile for the efficiency of triangular lattic.
float H_A; //(0.0)
float DD; //(0.0)
float DR; //(0.0)
float H_Q1x; //(0.0)
float H_Q1y; //(0.0)
float H_Q2x; //(0.0)
float H_Q2y; //(0.0)
//----- system variable setting end ------

//----- simulation setting ------
unsigned int BIN_SZ;
unsigned int BIN_NUM;
unsigned int EQUI_N;
unsigned int EQUI_Ni;
unsigned int relax_N;

float PTF = 0.1;             //Frequency of parallel tempering
char Output[128];  //set the output directory


void read_params(char* param_file){
  FILE* paramfp = fopen(param_file, "r");
  char tmp[128], readidx;
  readidx = fscanf(paramfp, "%s %d", tmp, &H_SpinSize);
  if ((readidx == -1)||(strcmp(tmp,"Size")!=0)){
    printf("read size error");
    exit(0);
  }
  if (H_SpinSize % 24 != 0){
    fprintf(stderr, "Please give a legal Size or revise cals.cu.\n");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %d", tmp, &H_SpinSize_z);
  if ((readidx == -1)||(strcmp(tmp,"Size_z")!=0)){
    printf("read size error");
    exit(0);
  }
  //if (H_SpinSize_z % 16 != 0){
  //  fprintf(stderr, "Please give a legal Size or revise cals.cu.\n");
  //  exit(0);
  //}
#ifdef THIN
#endif
#ifndef THIN
  //H_SpinSize_z = 1;
#endif
#ifdef TRI
  H_BlockSize_x = H_SpinSize / 3;
  H_BlockSize_y = H_SpinSize / 3;
  H_BlockSize_x = (H_BlockSize_x > 32)?(H_BlockSize_x/2):H_BlockSize_x;
  H_BlockSize_y = (H_BlockSize_y > 16)?(H_BlockSize_y/2):H_BlockSize_y;
  H_GridSize_x = H_SpinSize / H_BlockSize_x / 3;
  H_GridSize_y = H_SpinSize / H_BlockSize_y / 3;
#endif
#ifndef TRI
  H_BlockSize_x = H_SpinSize / 2;
  H_BlockSize_y = H_SpinSize / 2;
  H_BlockSize_x = (H_BlockSize_x > 32)?32:H_BlockSize_x;
  H_BlockSize_y = (H_BlockSize_y > 16)?16:H_BlockSize_y;
  H_GridSize_x = H_SpinSize / H_BlockSize_x / 2;
  H_GridSize_y = H_SpinSize / H_BlockSize_y / 2;
#endif
  H_N = H_SpinSize * H_SpinSize * H_SpinSize_z;
  H_Nplane = H_SpinSize * H_SpinSize;
  H_TN = H_BlockSize_x * H_BlockSize_y;
  H_BN = H_GridSize_x * H_GridSize_y;
  block = H_BlockSize_x * H_BlockSize_y;
  printf("%d\n", block);
  fflush(stdout);
  caloutputsize = block * sizeof(double);
  rngShmemsize = block * 4 * sizeof(unsigned);

  readidx = fscanf(paramfp, "%s %f", tmp, &H_A);
  if ((readidx == -1)||(strcmp(tmp,"A")!=0)){
    printf("read A error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %f", tmp, &DR);
  if ((readidx == -1)||(strcmp(tmp,"DR")!=0)){
    printf("read DR error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %f", tmp, &DD);
  if ((readidx == -1)||(strcmp(tmp,"DD")!=0)){
    printf("read DD error");
    exit(0);
  }
  H_Q1x = atan(sqrt((DD*DD+DR*DR)));//atan(sqrt((DD*DD+DR*DR)/2.0));
  H_Q1y = -0.5*atan(sqrt((DD*DD+DR*DR)));
  H_Q2x = 2*H_Q1x;
  H_Q2y = 2*H_Q1y;

  //----- system variable setting end ------

  //----- simulation setting ------
  readidx = fscanf(paramfp, "%s %d", tmp, &BIN_SZ);
  if ((readidx == -1)||(strcmp(tmp,"BIN_SIZE")!=0)){
    printf("read bin size error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %d", tmp, &BIN_NUM);
  if ((readidx == -1)||(strcmp(tmp,"BIN_NUM")!=0)){
    printf("read bin number error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %d", tmp, &EQUI_N);
  if ((readidx == -1)||(strcmp(tmp,"EQUI_N")!=0)){
    printf("read EQUI_N error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %d", tmp, &EQUI_Ni);
  if ((readidx == -1)||(strcmp(tmp,"EQUI_Ni")!=0)){
    printf("read EQUI_Ni error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %d", tmp, &relax_N);
  if ((readidx == -1)||(strcmp(tmp,"relax_N")!=0)){
    printf("read relax_N error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %f", tmp, &PTF);
  if ((readidx == -1)||(strcmp(tmp,"PTF")!=0)){
    printf("read parallel tempering frequency error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %d", tmp, &f_CORR);
  if ((readidx == -1)||(strcmp(tmp,"f_CORR")!=0)){
    printf("read f_CORR error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %d", tmp, &CORR_N);
  if ((readidx == -1)||(strcmp(tmp,"CORR_N")!=0)){
    printf("read CORR_N error");
    exit(0);
  }
  readidx = fscanf(paramfp, "%s %s", tmp, &Output);
  if ((readidx == -1)||(strcmp(tmp,"Output_dir")!=0)){
    printf("read output dir error");
    exit(0);
  }
  fclose(paramfp);
  //-- simulation setting end --
}
