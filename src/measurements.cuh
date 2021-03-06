#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H
#include "params.cuh"
#include "configuration.cuh"

#ifdef SQ
__global__ void cal2D(float *confx, float *confy, float *confz, double *out);
__global__ void getcorr2D(const float *confx, const float *confy, const float *confz, float *corr, int original_i, int original_j);
__global__ void sumcorr2D(double *DSum_corr, const float *corr, int *DTo);
__global__ void avgcorr2D(double *DSum_corr, double N_corr);
#endif
#ifdef TRI
__global__ void calTRI(float *confx, float *confy, float *confz, double *out);
__global__ void getcorrTRI(const float *confx, const float *confy, const float *confz, float *corr, int original_i, int original_j);
__global__ void sumcorrTRI(double *DSum_corr, const float *corr, int *DTo);
__global__ void avgcorrTRI(double *DSum_corr, double N_corr);
#endif
#ifdef THIN
__global__ void calthin(float *confx, float *confy, float *confz, double *out);
__global__ void getcorrthin(const float *confx, const float *confy, const float *confz, float *corr, int original_i, int original_j);
__global__ void avgcorrTHIN(double *DSum_corr, double N_corr);
__global__ void sumcorrTHIN(double *DSum_corr, const float *corr, int *DTo);
#endif
void move_params_device_cals();
void move_params_device_corr();


class measurement{
  public:
    measurement(char* indir, char* Oname, double normin, int Parallel_num);
    ~measurement();
    char fn[128];
    char dir[128];
    char name[128];
    FILE *fp;
    int data_num;
    unsigned int data_mem_size;
    double *outdata;
    double norm;
    void set_norm(int innorm);
    void normalize_and_save_and_reset();
};

class measurements{
  public:
    void* raw_memmory;
    unsigned int Out_mem_size;
    unsigned int Out_mem_size_s;
    measurements(char* indir, int Parallel_num, unsigned int binSize);
    ~measurements();
    int measurement_num;
    char names[20][10];
    double norms[20];
    int hist_start;
    int data_num;
    int data_num_s;
    std::vector<measurement> O;
    double *Hout;
    double **Dout;
    unsigned int *EHistogram;
    unsigned int *ChernHistogram;
    void measure(float** Dconfx, float** Dconfy, float** Dconfz, std::vector<int>& Ho, double* Ms, double* Es, float* HHs);
    void virtual_measure(float** Dconfx, float** Dconfy, float** Dconfz, std::vector<int>& Ho, double* Ms, double* Es, float* HHs);
    void normalize_and_save_and_reset();
    //void* raw_memmory;
};


class correlation{
  public:
    correlation(int Pnum, char* _Corrfn);
    ~correlation();
    int data_num;
    int data_num_s;
    unsigned int Spin_mem_size_s;
    unsigned int Spin_mem_size_p_s;
    unsigned int Spin_mem_size_d_s;
    unsigned int Spin_mem_size_d;
    int corrcount;
    double *HSum;
    float **Dcorr;
    double **DSum;
    char Corrfn[128];
    int Corrfd;
    int **DPo;//only use it only when extracting correlation
    void avg_write_reset(std::vector<int>& Ho);//set zero
    void extract(std::vector<int>& Ho, configuration &CONF);//==
    void changefile(char* _Corrfn);
    //write and set
};
#endif
