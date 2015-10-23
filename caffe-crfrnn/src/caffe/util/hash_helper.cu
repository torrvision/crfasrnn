#include <cstdlib>
#include <vector>
#include <cassert>
#include <cstdio>
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"


/*************************************************************/
/* Fast computation of modulo operator with constant divisor */
/*************************************************************/
/*__device__ __constant__ unsigned int __div_m;
__device__ __constant__ unsigned int __div_l;
__device__ __constant__ unsigned int __div_c;

#ifdef USE_CUSTOM_MODULO
__device__ inline unsigned int modHash(unsigned int n) {
  unsigned int t1 = __umulhi(__div_m, n);
  return n - ((t1+((n-t1)>>1))>>(__div_l-1)) * __div_c;
}
*/

namespace caffe {


#define modHash(n) ((n)%(2*table_capacity));


/*************************************************************/
/* End modulo                                                */
/*************************************************************/

//__device__ __constant__ unsigned int hOffset[64];

__device__ __host__ static unsigned int hash(signed short *key, int kd) {
    unsigned int k = 0; 
    for (int i = 0; i < kd; i++) {
	k += key[i];
	k = k * 2531011; 
    }
    return k;
}

__device__ __host__ static unsigned int hash(int *key, int kd) {
    unsigned int k = 0; 
    for (int i = 0; i < kd; i++) {
	k += key[i];
	k = k * 2531011; 
    }
    return k;
}
/*
template<int d> __device__ static bool matchKey(int idx, signed short * key) {
    bool match = true;
    int slot = idx/(d+1), color = idx-slot*(d+1);
    char *rank = table_rank + slot * (d+1);
    signed short *zero = table_zeros + slot * (d+1);
    for (int i = 0; i < d && match; i++) {
	match = (key[i] == zero[i] + color - (rank[i] > d-color ? (d+1) : 0));
    }
    return match;
}
*/

/*
static float* swapHashTableValues(float *newValues) {
    float * oldValues;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&oldValues,
					table_values,
					sizeof(float *)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_values,
				      &newValues,
				      sizeof(float *)));
    return oldValues;
}
*/

__device__ static int hashTableInsert(unsigned int fh, signed short *key,
    signed short* table_keys,
    int* table_entries,
    int table_capacity, 
    unsigned int slot, 
    int kd) 
{    	
    int h = modHash(fh);
    while (1) {
	int *e = &table_entries[h];

	// If the cell is empty (-1), lock it (-2)
	int contents = atomicCAS(e, -1, -2);

	if (contents == -2) {
	    // If it was locked already, move on to the next cell

	} else if (contents == -1) { 
	    // If it was empty, we successfully locked it. Write our key.

	    for (int i = 0; i < kd; i++) {
		table_keys[slot*kd+i] = key[i];
	    }

	    // Unlock
	    atomicExch(e, slot); 

	    return h;
	} else {
	    // The cell is unlocked and has a key in it, check if it matches
          //  #ifdef LINEAR_D_MEMORY
 	    //if (matchKey<kd>(contents, key)) return h;
          //  #else
	    bool match = true;
	    for (int i = 0; i < kd && match; i++) {
		match = (table_keys[contents*kd+i] == key[i]);
	    }
	    if (match) return h;
          //  #endif       

	}
	// increment the bucket with wraparound
	h++;
	if (h == table_capacity*2) h = 0;
    }
}

__device__ static int hashTableInsert(signed short *key, 
    signed short* table_keys,
    int* table_entries,
    int table_capacity, 
    unsigned int slot, 
    int kd) 
{
    unsigned int myHash = hash(key, kd);
    return hashTableInsert(myHash, key, table_keys, table_entries, table_capacity, slot, kd);
}


/*
template<int kd> __device__ static
int hashTableRetrieveWithHash(unsigned int fh, signed short *key) {
  int h = modHash(fh);
  while (1) {
    int *e = table_entries + h;
    
    if (*e == -1) return -1;
    
    #ifdef LINEAR_D_MEMORY
    if (matchKey<kd>((*e), key)) return *e;
    #else
    bool match = true;
    for (int i = 0; i < kd && match; i++) {
      match = (table_keys[(*e)*kd+i] == key[i]);
    }
    if (match) return *e;
    #endif
    
    h++;
    if (h == table_capacity*2) h = 0;
  }
}
   */

__device__ static int hashTableRetrieve(signed short *key,
	  int* table_entries,
	  signed short* table_keys,
	  int table_capacity,
        int kd) 
{
    int h = modHash(hash(key, kd));
    while (1) {
	int *e = table_entries + h;

	if (*e == -1) return -1;

	bool match = true;
	for (int i = 0; i < kd && match; i++) {
	    match = (table_keys[(*e)*kd+i] == key[i]);
	}
	if (match) return *e;

	h++;
	if (h == table_capacity*2) h = 0;
    }
}

} //namespace caffe
