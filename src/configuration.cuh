#ifndef CONFIGURATION_H
#define CONFIGURATION_H
#include "params.cuh"

class configuration{
	public:
		configuration(int Pnum, char* conf_dir);
		~configuration();
		unsigned int Spin_mem_size;
		unsigned int configurations_num;
		unsigned int spins_num;
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
		void writedata();
		int Confxfd;
		int Confyfd;
		int Confzfd;
};
#endif
