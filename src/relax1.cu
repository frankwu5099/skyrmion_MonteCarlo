//-----Spin-flip for top-right corner (0,0)                      //and bottom-left corner-----
__global__ void relaxTLBR(float *confx, float *confy, float *confz, unsigned int *rngState, float H){
	//Energy variables
	float phix, phiy;
	float sx, sy, sz;
	float s2x, s2y, s2z;
	float hx, hy, hz;
	float Az, Xxy;
	float norm;
	float dot;
	const int x = threadIdx.x % BlockSize_x;
	const int y = threadIdx.x / BlockSize_x;
	const int tx = 2 * (((blockIdx.x % BN) % GridSize_x) * BlockSize_x + x);
	const int ty =(blockIdx.x / BN) * SpinSize +  2 * ((((blockIdx.x % BN) / GridSize_x) % GridSize_y) * BlockSize_y + y);
	int ib, jb, i, j;
	//----------Spin flip at the top and left corner of each thread sqare----------
	i = tx;
	j = ty;
	ib = (i + SpinSize - 1) % SpinSize;
	if((j % SpinSize) == 0)	jb = j + SpinSize - 1;
	else			jb = j - 1;
	//Spin flip!
	hx = BXPxx(bd) * confx[j * SpinSize + i+1] + BYPxx(bd) * confx[(j+1) * SpinSize + i] + BXMxx(bd) * confx[j * SpinSize + ib] + BYMxx(bd) * confx[jb * SpinSize + i]\
	   + BXPxy(bd) * confy[j * SpinSize + i+1] + BYPxy(bd) * confy[(j+1) * SpinSize + i] + BXMxy(bd) * confy[j * SpinSize + ib] + BYMxy(bd) * confy[jb * SpinSize + i]\
	   + BXPxz(bd) * confz[j * SpinSize + i+1] + BYPxz(bd) * confz[(j+1) * SpinSize + i] + BXMxz(bd) * confz[j * SpinSize + ib] + BYMxz(bd) * confz[jb * SpinSize + i];
	hy = BXPyx(bd) * confx[j * SpinSize + i+1] + BYPyx(bd) * confx[(j+1) * SpinSize + i] + BXMyx(bd) * confx[j * SpinSize + ib] + BYMyx(bd) * confx[jb * SpinSize + i]\
	   + BXPyy(bd) * confy[j * SpinSize + i+1] + BYPyy(bd) * confy[(j+1) * SpinSize + i] + BXMyy(bd) * confy[j * SpinSize + ib] + BYMyy(bd) * confy[jb * SpinSize + i]\
	   + BXPyz(bd) * confz[j * SpinSize + i+1] + BYPyz(bd) * confz[(j+1) * SpinSize + i] + BXMyz(bd) * confz[j * SpinSize + ib] + BYMyz(bd) * confz[jb * SpinSize + i];
	hz = BXPzx(bd) * confx[j * SpinSize + i+1] + BYPzx(bd) * confx[(j+1) * SpinSize + i] + BXMzx(bd) * confx[j * SpinSize + ib] + BYMzx(bd) * confx[jb * SpinSize + i]\
	   + BXPzy(bd) * confy[j * SpinSize + i+1] + BYPzy(bd) * confy[(j+1) * SpinSize + i] + BXMzy(bd) * confy[j * SpinSize + ib] + BYMzy(bd) * confy[jb * SpinSize + i]\
	   + BXPzz(bd) * confz[j * SpinSize + i+1] + BYPzz(bd) * confz[(j+1) * SpinSize + i] + BXMzz(bd) * confz[j * SpinSize + ib] + BYMzz(bd) * confz[jb * SpinSize + i] + H;
	if(hx * hx + hy * hy + hz * hz>0.001){
	norm = 1 / sqrt(hx * hx + hy * hy + hz * hz);
	hx *= norm;
	hy *= norm;
	hz *= norm;
	sx = confx[j*SpinSize + i];
	sy = confy[j*SpinSize + i];
	sz = confz[j*SpinSize + i];
	dot = hx * sx + hy * sy + hz * sz;
	Xxy = sx * hy - sy * hx;
	Az = sz - dot * hz;
	phix = (Az*Az + Xxy*Xxy >0.00001 && (Xxy*Xxy>0.0000001))?((Az*Az - Xxy*Xxy)/(Az*Az + Xxy*Xxy)):1.0;
	phiy = (Az*Az + Xxy*Xxy >0.00001 && (Xxy*Xxy>0.0000001))?((1-phix)*Az/Xxy):0.0;
	dot = dot * (1 - phix);
	s2x = sx * phix + hx * dot + (sy * hz - sz * hy) * phiy;
	s2y = sy * phix + hy * dot + (sz * hx - sx * hz) * phiy;
	s2z = sz * phix + hz * dot + Xxy * phiy;
	norm = 1 / sqrt(s2x * s2x + s2y * s2y + s2z * s2z);
	confx[j*SpinSize + i] = s2x * norm;
	confy[j*SpinSize + i] = s2y * norm;
	confz[j*SpinSize + i] = s2z * norm;
	}//exception
	__syncthreads();
	//----------Spin flip at the bottom and right corner of each thread sqare----------
	i = tx + 1;
	j = ty + 1;
	ib = (i + 1) % SpinSize;
	if((j % SpinSize) == SpinSize - 1)	jb = j - SpinSize + 1;
	else					jb = j + 1;
	//Spin flip!
	hx = BXPxx(bd) * confx[j * SpinSize + ib] + BYPxx(bd) * confx[jb * SpinSize + i] + BXMxx(bd) * confx[j * SpinSize + i-1] + BYMxx(bd) * confx[(j-1) * SpinSize + i]\
	   + BXPxy(bd) * confy[j * SpinSize + ib] + BYPxy(bd) * confy[jb * SpinSize + i] + BXMxy(bd) * confy[j * SpinSize + i-1] + BYMxy(bd) * confy[(j-1) * SpinSize + i]\
	   + BXPxz(bd) * confz[j * SpinSize + ib] + BYPxz(bd) * confz[jb * SpinSize + i] + BXMxz(bd) * confz[j * SpinSize + i-1] + BYMxz(bd) * confz[(j-1) * SpinSize + i];
	hy = BXPyx(bd) * confx[j * SpinSize + ib] + BYPyx(bd) * confx[jb * SpinSize + i] + BXMyx(bd) * confx[j * SpinSize + i-1] + BYMyx(bd) * confx[(j-1) * SpinSize + i]\
	   + BXPyy(bd) * confy[j * SpinSize + ib] + BYPyy(bd) * confy[jb * SpinSize + i] + BXMyy(bd) * confy[j * SpinSize + i-1] + BYMyy(bd) * confy[(j-1) * SpinSize + i]\
	   + BXPyz(bd) * confz[j * SpinSize + ib] + BYPyz(bd) * confz[jb * SpinSize + i] + BXMyz(bd) * confz[j * SpinSize + i-1] + BYMyz(bd) * confz[(j-1) * SpinSize + i];
	hz = BXPzx(bd) * confx[j * SpinSize + ib] + BYPzx(bd) * confx[jb * SpinSize + i] + BXMzx(bd) * confx[j * SpinSize + i-1] + BYMzx(bd) * confx[(j-1) * SpinSize + i]\
	   + BXPzy(bd) * confy[j * SpinSize + ib] + BYPzy(bd) * confy[jb * SpinSize + i] + BXMzy(bd) * confy[j * SpinSize + i-1] + BYMzy(bd) * confy[(j-1) * SpinSize + i]\
	   + BXPzz(bd) * confz[j * SpinSize + ib] + BYPzz(bd) * confz[jb * SpinSize + i] + BXMzz(bd) * confz[j * SpinSize + i-1] + BYMzz(bd) * confz[(j-1) * SpinSize + i] + H;
	if(hx * hx + hy * hy + hz * hz>0.001){
	norm = 1 / sqrt(hx * hx + hy * hy + hz * hz);
	hx *= norm;
	hy *= norm;
	hz *= norm;
	sx = confx[j*SpinSize + i];
	sy = confy[j*SpinSize + i];
	sz = confz[j*SpinSize + i];
	dot = hx * sx + hy * sy + hz * sz;
	Xxy = sx * hy - sy * hx;
	Az = sz - dot * hz;
	phix = (Az*Az + Xxy*Xxy >0.00001 && (Xxy*Xxy>0.0000001))?((Az*Az - Xxy*Xxy)/(Az*Az + Xxy*Xxy)):1.0;
	phiy = (Az*Az + Xxy*Xxy >0.00001 && (Xxy*Xxy>0.0000001))?((1-phix)*Az/Xxy):0.0;
	dot = dot * (1 - phix);
	s2x = sx * phix + hx * dot + (sy * hz - sz * hy) * phiy;
	s2y = sy * phix + hy * dot + (sz * hx - sx * hz) * phiy;
	s2z = sz * phix + hz * dot + Xxy * phiy;
	norm = 1 / sqrt(s2x * s2x + s2y * s2y + s2z * s2z);
	confx[j*SpinSize + i] = s2x * norm;
	confy[j*SpinSize + i] = s2y * norm;
	confz[j*SpinSize + i] = s2z * norm;
	}//exception
	__syncthreads();
	//Load random number back to global memory
}
