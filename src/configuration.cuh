#ifndef PARAMS_H
#define PARAMS_H
#include "params.cuh"
#endif

class configuration{
	public:
		unsigned int Spin_mem_size;
		unsigned int configurations_num;
		unsigned int spins_num;
		configuration(int Pnum, char* conf_dir);
		char Confxfn[128];
		char Confyfn[128];
		char Confzfn[128];
		float *Hx;
		float *Hy;
		float *Hz;
		float *Dx;
		float *Dy;
		float *Dz;
		void initialize (bool order);
		void backtoHost();
		void write();
		int Confxfd;
		int Confyfd;
		int Confzfd;

};


