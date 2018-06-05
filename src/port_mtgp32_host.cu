#include "port_mtgp32_host.h"


void mtgp32_init_state(unsigned int state[],
              const mtgp32_params_fast_t *para, unsigned int seed) {
    int i;
    int size = para->mexp / 32 + 1;
    unsigned int hidden_seed;
    unsigned int tmp;
    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(state, tmp & 0xff, sizeof(unsigned int) * size);
    state[0] = seed;
    state[1] = hidden_seed;
    for (i = 1; i < size; i++) {
    state[i] ^= (1812433253) * (state[i - 1]
                        ^ (state[i - 1] >> 30))
        + i;
    }
}


/*
 * This function initializes the internal state array
 * with a 32-bit integer array. \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 *
 * @param[out] mtgp32 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a 32-bit integer array used as a seed.
 * @param[in] length length of the array.
 * @return CURAND_STATUS_SUCCESS 
 */

int mtgp32_init_by_array(unsigned int state[],
             const mtgp32_params_fast_t *para,
             unsigned int *array, int length) {
    int i, j, count;
    unsigned int r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    unsigned int hidden_seed;
    unsigned int tmp;

    if (size >= 623) {
    lag = 11;
    } else if (size >= 68) {
    lag = 7;
    } else if (size >= 39) {
    lag = 5;
    } else {
    lag = 3;
    }
    mid = (size - lag) / 2;

    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(state, tmp & 0xff, sizeof(unsigned int) * size);
    state[0] = hidden_seed;

    if (length + 1 > size) {
    count = length + 1;
    } else {
    count = size;
    }
    r = ini_func1(state[0] ^ state[mid] ^ state[size - 1]);
    state[mid] += r;
    r += length;
    state[(mid + lag) % size] += r;
    state[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += array[j] + i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (; j < count; j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
    r = ini_func2(state[i] + state[(i + mid) % size]
              + state[(i + size - 1) % size]);
    state[(i + mid) % size] ^= r;
    r -= i;
    state[(i + mid + lag) % size] ^= r;
    state[i] = r;
    i = (i + 1) % size;
    }
    if (state[size - 1] == 0) {
    state[size - 1] = non_zero;
    }
    return 0;
}

/*
 * This function initializes the internal state array
 * with a character array. \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 * This is the same algorithm with mtgp32_init_by_array(), but hope to
 * be more useful.
 *
 * @param[out] mtgp32 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a character array used as a seed. (terminated by zero.)
 * @return memory allocation result. if 0 then O.K.
 */
int mtgp32_init_by_str(unsigned int state[],
               const mtgp32_params_fast_t *para, unsigned char *array) {
    int i, j, count;
    unsigned int r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    int length = (unsigned int)strlen((char *)array);
    unsigned int hidden_seed;
    unsigned int tmp;

    if (size >= 623) {
    lag = 11;
    } else if (size >= 68) {
    lag = 7;
    } else if (size >= 39) {
    lag = 5;
    } else {
    lag = 3;
    }
    mid = (size - lag) / 2;

    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(state, tmp & 0xff, sizeof(unsigned int) * size);
    state[0] = hidden_seed;

    if (length + 1 > size) {
    count = length + 1;
    } else {
    count = size;
    }
    r = ini_func1(state[0] ^ state[mid] ^ state[size - 1]);
    state[mid] += r;
    r += length;
    state[(mid + lag) % size] += r;
    state[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += array[j] + i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (; j < count; j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
    r = ini_func2(state[i] + state[(i + mid) % size]
              + state[(i + size - 1) % size]);
    state[(i + mid) % size] ^= r;
    r -= i;
    state[(i + mid + lag) % size] ^= r;
    state[i] = r;
    i = (i + 1) % size;
    }
    if (state[size - 1] == 0) {
    state[size - 1] = non_zero;
    }
    return 0;
}



/**
 * \brief Set up constant parameters for the mtgp32 generator
 *
 * This host-side helper function re-organizes CURAND_NUM_MTGP32_PARAMS sets of 
 * generator parameters for use by kernel functions and copies the 
 * result to the specified location in device memory.
 *
 * \param params - Pointer to an array of type mtgp32_params_fast_t in host memory
 * \param p - pointer to a structure of type mtgp32_kernel_params_t in device memory.
 *
 * \return 
 * - CURAND_STATUS_ALLOCATION_FAILED if host memory could not be allocated
 * - CURAND_STATUS_INITIALIZATION_FAILED if the copy to device memory failed
 * - CURAND_STATUS_SUCCESS otherwise
 */
__host__ curandStatus_t curandMakeMTGP32Constants(const mtgp32_params_fast_t params[], mtgp32_kernel_params_t *& p) {
    const int block_num = CURAND_NUM_MTGP32_PARAMS;
    const int size1 = sizeof(unsigned int) * block_num;
    const int size2 = sizeof(unsigned int) * block_num * TBL_SIZE;
    unsigned int *h_pos_tbl;
    unsigned int *h_sh1_tbl;
    unsigned int *h_sh2_tbl;
    unsigned int *h_param_tbl;
    unsigned int *h_temper_tbl;
    unsigned int *h_single_temper_tbl;
    unsigned int *h_mask;
    curandStatus_t status = CURAND_STATUS_SUCCESS;
    
    h_pos_tbl = (unsigned int *)malloc(size1);
    h_sh1_tbl = (unsigned int *)malloc(size1);
    h_sh2_tbl = (unsigned int *)malloc(size1);
    h_param_tbl = (unsigned int *)malloc(size2);
    h_temper_tbl = (unsigned int *)malloc(size2);
    h_single_temper_tbl = (unsigned int *)malloc(size2);
    h_mask = (unsigned int *)malloc(sizeof(unsigned int));
    if (h_pos_tbl == NULL
	    || h_sh1_tbl == NULL
	    || h_sh2_tbl == NULL
	    || h_param_tbl == NULL
	    || h_temper_tbl == NULL
	    || h_single_temper_tbl == NULL
	    || h_mask == NULL) {
        if (h_pos_tbl != NULL) free(h_pos_tbl);
        if (h_sh1_tbl != NULL) free(h_sh1_tbl);
        if (h_sh2_tbl != NULL) free(h_sh2_tbl);
        if (h_param_tbl != NULL) free(h_param_tbl);
        if (h_temper_tbl != NULL) free(h_temper_tbl);
        if (h_single_temper_tbl != NULL) free(h_single_temper_tbl);
        if (h_mask != NULL) free(h_mask);
        status = CURAND_STATUS_ALLOCATION_FAILED;
    } else {       

        h_mask[0] = params[0].mask;
        for (int i = 0; i < block_num; i++) {
	        h_pos_tbl[i] = params[i].pos;
	        h_sh1_tbl[i] = params[i].sh1;
	        h_sh2_tbl[i] = params[i].sh2;
	        for (int j = 0; j < TBL_SIZE; j++) {
	            h_param_tbl[i * TBL_SIZE + j] = params[i].tbl[j];
	            h_temper_tbl[i * TBL_SIZE + j] = params[i].tmp_tbl[j];
	            h_single_temper_tbl[i * TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
	        }
        }
        if (cudaMemcpy( p->pos_tbl, 
                        h_pos_tbl, size1, cudaMemcpyHostToDevice) != cudaSuccess)
        { 
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->sh1_tbl, 
                        h_sh1_tbl, size1, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->sh2_tbl, 
                        h_sh2_tbl, size1, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->param_tbl, 
                        h_param_tbl, size2, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->temper_tbl, 
                        h_temper_tbl, size2, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->single_temper_tbl, 
                        h_single_temper_tbl, size2, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->mask, 
                        h_mask, sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } 
    }
    if (h_pos_tbl != NULL) free(h_pos_tbl);
    if (h_sh1_tbl != NULL) free(h_sh1_tbl);
    if (h_sh2_tbl != NULL) free(h_sh2_tbl);
    if (h_param_tbl != NULL) free(h_param_tbl);
    if (h_temper_tbl != NULL) free(h_temper_tbl);
    if (h_single_temper_tbl != NULL)free(h_single_temper_tbl);
    if (h_mask != NULL) free(h_mask);
    return status;
}

/**
 * \brief Set up initial states for the mtgp32 generator
 *
 * This host-side helper function initializes a number of states (one parameter set per state) for 
 * an mtgp32 generator. To accomplish this it allocates a state array in host memory,
 * initializes that array, and copies the result to device memory.
 *
 V* \param s - pointer to an array of states in device memory
 * \param params - Pointer to an array of type mtgp32_params_fast_t in host memory
 * \param k - pointer to a structure of type mtgp32_kernel_params_t in device memory
 * \param n - number of parameter sets/states to initialize
 * \param seed - seed value
 *
 * \return 
 * - CURAND_STATUS_ALLOCATION_FAILED if host memory state could not be allocated 
 * - CURAND_STATUS_INITIALIZATION_FAILED if the copy to device memory failed
 * - CURAND_STATUS_SUCCESS otherwise
 */
__host__ curandStatus_t CURANDAPI curandMakeMTGP32KernelState(curandStateMtgp32_t *s,
                                            mtgp32_params_fast_t params[],
                                            mtgp32_kernel_params_t *k,
                                            int n,
                                            unsigned long long seed)
{
    int i;
    curandStatus_t status = CURAND_STATUS_SUCCESS;
    curandStateMtgp32_t *h_status =(curandStateMtgp32_t *) malloc(sizeof(curandStateMtgp32_t) * n);
    if (h_status == NULL) {
        status = CURAND_STATUS_ALLOCATION_FAILED;
    } else {
        seed = seed ^ (seed >> 32);
        for (i = 0; i < n; i++) {
            mtgp32_init_state(&(h_status[i].s[0]), &params[i],(unsigned int)seed + i + 1);
            h_status[i].offset = 0;
            h_status[i].pIdx = i;
            h_status[i].k = k;
            h_status[i].precise_double_flag = 0;
        }
        if (cudaMemcpy(s, h_status,
                       sizeof(curandStateMtgp32_t) * n,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        }
     }
    free(h_status);
    return status;
}



