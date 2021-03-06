#ifndef CONFIGURATION_H
#define CONFIGURATION_H
#include "params.cuh"

class configuration{
	public:
		configuration(int Pnum, char* conf_dir);
		~configuration();
		unsigned int Spin_mem_size;
		unsigned int Spin_mem_size_s;
		unsigned int Single_mem_size;
		unsigned int configurations_num;
		unsigned int configurations_num_s;
		unsigned int spins_num;
		unsigned int spins_num_s;
		unsigned int f_index;
		char dirfn[128];
		char Confxfn[128];
		char Confyfn[128];
		char Confzfn[128];
		float *Hx;
		float *Hy;
		float *Hz;
		float **Dx;
		float **Dy;
		float **Dz;
		void initialize (bool order, mt19937 &generator);
		void backtoHost();
		void writedata();
		void Dominatestateback(int hostid, int deviceid);
		int Confxfd;
		int Confyfd;
		int Confzfd;
};
#endif
