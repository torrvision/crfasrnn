#define BLOCK_SIZE 64

#define _DEBUG
#include <stdio.h>
#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/hash_helper.cu"

namespace caffe {

static void swapHashTableValues(float* oldValues, float *newValues, float* table_values,size_t size) {
  CUDA_CHECK(cudaMemcpy(oldValues,table_values,size,cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(table_values,newValues,size,cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(newValues,oldValues,size,cudaMemcpyDeviceToDevice));
}

template<int pd>
__global__ static void createMatrix(const int w, const int h,
				    const float *positions,
				    int *table_entries,
				    int table_capacity,
				    signed short* table_keys,				    
				    const float *scaleFactor,
				    MatrixEntry *matrix)
{
    // scanline order
    //const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    //const bool outOfBounds = (idx>=num_points) ;
    //const int threadId = idx;

    // 8x8 blocks
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    const int idx = y*w + x;
    const bool outOfBounds = (x >= w) || (y >= h);
  
    float myElevated[pd+1];
    const float *myPosition = positions + idx*pd;

    int myGreedy[pd+1];
    int myRank[pd+1];

    float myBarycentric[pd+2];
    __shared__ short keys[pd*BLOCK_SIZE];
    short *myKey = keys + threadId * pd;

    if (!outOfBounds) {

	myElevated[pd] = -pd*(myPosition[pd-1])*scaleFactor[pd-1];
	for (int i = pd-1; i > 0; i--) {
	    myElevated[i] = (myElevated[i+1] -
			     i*(myPosition[i-1])*scaleFactor[i-1] +
			     (i+2)*(myPosition[i])*scaleFactor[i]);
	}
	myElevated[0] = myElevated[1] + 2*(myPosition[0])*scaleFactor[0];


	// find the closest zero-colored lattice point

	// greedily search for the closest zero-colored lattice point
	signed short sum = 0;
	for (int i = 0; i <= pd; i++) {
	    float v = myElevated[i]*(1.0f/(pd+1));
	    float up = ceilf(v) * (pd+1);
	    float down = floorf(v) * (pd+1);
	    if (up - myElevated[i] < myElevated[i] - down) {
		myGreedy[i] = (signed short)up;
	    } else {
		myGreedy[i] = (signed short)down;
	    }
	    sum += myGreedy[i];
	}
	sum /= pd+1;

	// sort differential to find the permutation between this simplex and the canonical one
	for (int i = 0; i <= pd; i++) {
	    myRank[i] = 0;
	    for (int j = 0; j <= pd; j++) {
		if (myElevated[i] - myGreedy[i] < myElevated[j] - myGreedy[j] ||
		    (myElevated[i] - myGreedy[i] == myElevated[j] - myGreedy[j]
		     && i > j)) {
		    myRank[i]++;
		}
	    }
	}

	if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
	    for (int i = 0; i <= pd; i++) {
		if (myRank[i] >= pd + 1 - sum) {
		    myGreedy[i] -= pd+1;
		    myRank[i] += sum - (pd+1);
		} else {
		    myRank[i] += sum;
		}
	    }
	} else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
	    for (int i = 0; i <= pd; i++) {
		if (myRank[i] < -sum) {
		    myGreedy[i] += pd+1;
		    myRank[i] += (pd+1) + sum;
		} else {
		    myRank[i] += sum;
		}
	    }
	}

	// turn delta into barycentric coords
	for (int i = 0; i <= pd+1; i++) {
	    myBarycentric[i] = 0;
	}

	for (int i = 0; i <= pd; i++) {
	    float delta = (myElevated[i] - myGreedy[i]) * (1.0f/(pd+1));
	    myBarycentric[pd-myRank[i]] += delta;
	    myBarycentric[pd+1-myRank[i]] -= delta;
	}
	myBarycentric[0] += 1.0f + myBarycentric[pd+1];
    }

    #ifdef USE_ADDITIVE_HASH
    unsigned int cumulative_hash = hash<pd>(myGreedy);
    #endif
    for (int color = 0; color <= pd; color++) {
	// Compute the location of the lattice point explicitly (all but
	// the last coordinate - it's redundant because they sum to zero)
	if (!outOfBounds) {
	    for (int i = 0; i < pd; i++) {
		myKey[i] = myGreedy[i] + color;
		if (myRank[i] > pd-color) myKey[i] -= (pd+1);
	    }
	}

	#ifdef USE_ADDITIVE_HASH
	for (int i = 0; i < pd; i++) {
	    if (myRank[i] == pd-color) cumulative_hash += hOffset[i];
	}
	#endif

	if (!outOfBounds) {
	    MatrixEntry r;
	    #ifdef USE_ADDITIVE_HASH
	    r.index = hashTableInsert<pd>(cumulative_hash, myKey, table_keys,
    		table_entries, table_capacity,  idx*(pd+1)+color);
	    #else
	    r.index = hashTableInsert<pd>(myKey, table_keys, table_entries,
    		table_capacity,  idx*(pd+1)+color);
	    #endif
	    r.weight = myBarycentric[color];
	    matrix[idx*(pd+1) + color] = r;
	}
    }
}

template<int kd>
__global__ static void cleanHashTable(const int n,
				    int *table_entries,
				    int table_capacity,
				    signed short* table_keys)
{
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n) return;

    // find my hash table entry
    int *e = table_entries + idx;

    // Check if I created my own key in the previous phase
    if (*e >= 0) {
	// Rehash my key and reset the pointer in order to merge with
	// any other pixel that created a different entry under the
	// same key. If the computation was serial this would never
	// happen, but sometimes race conditions can make the same key
	// be inserted twice. hashTableRetrieve always returns the
	// earlier, so it's no problem as long as we rehash now.
	*e = hashTableRetrieve<kd>(table_keys + *e*kd,
	        table_entries, table_keys, table_capacity);
    }
}

template<int pd>
__global__ static void resetIndex(const int w, const int h,
  MatrixEntry *matrix,
  int *table_entries) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + (blockIdx.y/(pd+1)) * blockDim.y;
  const int color = blockIdx.y % (pd+1);
  const int idx = y*w + x;
  const bool outOfBounds = (x >= w) || (y >= h); 
  if (!outOfBounds){
    MatrixEntry r = matrix[idx*(pd+1)+color];
    matrix[idx*(pd+1)+color].index = table_entries[r.index];
  }
}

template<int pd, int vd>
__global__ static void splatCache(const int w, const int h,
		 const float *values,
		 const MatrixEntry *matrix,
		 const int *table_entries,
		 float *table_values) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + (blockIdx.y/(pd+1)) * blockDim.y;
    const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    const int color = blockIdx.y % (pd+1);
    const int idx = y*w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    __shared__ int sharedOffsets[BLOCK_SIZE];
    __shared__ float sharedValues[BLOCK_SIZE*(vd+1)];
    int myOffset = -1;
    float *myValue = sharedValues + threadId*(vd+1);

    if (!outOfBounds) {

	const float *value = values + idx;

	MatrixEntry r = matrix[idx*(pd+1)+color];

	// convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
	//matrix[idx*(pd+1)+color].index = r.index = table_entries[r.index];
	// record the offset into the keys/values array in shared space
	myOffset = sharedOffsets[threadId] = r.index*(vd+1);

	for (int j = 0; j < vd; j++) {
	    myValue[j] = value[j*w*h]*r.weight;
	}
	myValue[vd] = r.weight;

    } else {
	sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    // am I the first thread in this block to care about this key?

    if (outOfBounds) return;

    for (int i = 0; i < BLOCK_SIZE; i++) {
	if (i < threadId) {
	    if (myOffset == sharedOffsets[i]) {
		// somebody else with higher priority cares about this key
		return;
	    }
	} else if (i > threadId) {
	    if (myOffset == sharedOffsets[i]) {
		// someone else with lower priority cares about this key, accumulate it into mine
		for (int j = 0; j <= vd; j++) {
		    sharedValues[threadId*(vd+1) + j] += sharedValues[i*(vd+1) + j];
		}
	    }
	}
    }

    // only the threads with something to write to main memory are still going
    float *val = table_values + myOffset;
    for (int j = 0; j <= vd; j++) {
	atomicAdd(val+j, myValue[j]);
    }
}

template<int pd, int vd>
__global__ static void blur(int n, float *newValues, 
   const MatrixEntry *matrix,
   const int *table_entries,
   const signed short *table_keys,
   const int table_capacity,
   float *table_values,
   int color) {
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n) return;

    // Check if I'm valid
    if (matrix[idx].index != idx) return;

    // find my key and the keys of my neighbours
    short myKey[pd+1];
    short np[pd+1];
    short nm[pd+1];

    for (int i = 0; i < pd; i++) {
        myKey[i] = table_keys[idx*pd+i];
	np[i] = myKey[i]+1;
	nm[i] = myKey[i]-1;
    }


    np[color] -= pd+1;
    nm[color] += pd+1;

#ifdef USE_ADDITIVE_HASH
    unsigned int hCurrent = hash<pd>(myKey);
    int offNp = hashTableRetrieveWithHash<pd>(hCurrent+hOffset[color],np, table_entries, table_keys, table_capacity);
    int offNm = hashTableRetrieveWithHash<pd>(hCurrent-hOffset[color],nm, table_entries, table_keys, table_capacity);
#else
    int offNp = hashTableRetrieve<pd>(np, table_entries, table_keys, table_capacity);
    int offNm = hashTableRetrieve<pd>(nm, table_entries, table_keys, table_capacity);
#endif

    float *valMe = table_values + (vd+1)*idx;
    float *valNp = table_values + (vd+1)*offNp;
    float *valNm = table_values + (vd+1)*offNm;
    float *valOut = newValues + (vd+1)*idx;

    if (offNp >= 0 && offNm >= 0) {
	for (int i = 0; i <= vd; i++) {
	    valOut[i] = (valNp[i] + (valMe[i]*2) + valNm[i])/4;
	}
    } else if (offNp >= 0) {
	for (int i = 0; i <= vd; i++) {
	    valOut[i] = (valNp[i] + (valMe[i]*2))/4;
	}
    } else if (offNm >= 0) {
	for (int i = 0; i <= vd; i++) {
	    valOut[i] = (valNm[i] + (valMe[i]*2))/4;
	}
    } else {
	for (int i = 0; i <= vd; i++) {
	    valOut[i] = valMe[i]*2;
	}
    }
}

template<int pd, int vd>
__global__ static void slice(const int w, const int h,
  float *values, 
  const MatrixEntry *matrix, 
  float *table_values,
  bool add) {
    //const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    const int idx = y*w + x;
    const bool outOfBounds = (x >= w) || (y >= h);

    if (outOfBounds) return;

    __shared__ float localValue[BLOCK_SIZE*vd];

    float *myValue = localValue + threadId*vd;
    float myWeight = 0;

    for (int i = 0; i < vd; i++) {
	myValue[i] = 0;
    }

    for (int i = 0; i <= pd; i++) {
	MatrixEntry r = matrix[idx*(pd+1) + i];
	const float *val = table_values + r.index*(vd+1);
	for (int j = 0; j < vd; j++) {
	    myValue[j] += r.weight*val[j];
	}
	myWeight += r.weight*val[vd];
    }

    myWeight = 1.0f/myWeight;
    float alpha = 1.0f;// / (1+powf(2, -pd));
    for (int j = 0; j < vd; j++){
      if(!add)
        values[j*w*h + idx] = 0;
	values[j*w*h + idx] += myValue[j]*alpha*myWeight;
    }
}


template<int pd>
void gpu_init(const float* features,
    HashTable* table, 
    MatrixEntry* matrix, 
    const int w, const int h)
{
    int num_points = w*h ;
    // Scan line order
    //unsigned int blocks = (num_points-1)/64 + 1;
    //unsigned int blockSize = 64;
    dim3 blocks((w-1)/8+1, (h-1)/8+1, 1);
    dim3 blockSize(8, 8, 1);
    
    float blurVariance = 0.5 ;
    float * scaleFactor;
    float* scaleFactorHost = new float[pd];
    
    // Create Scale factor vector and give it to GPU
    // num_dimensions is likely to be low so do that 
    // on the CPU
    for (int i = 0; i < pd; i++) {
	scaleFactorHost[i] = (pd+1)*sqrtf((1.0/6 + blurVariance)/((i+1)*(i+2)));
    }
    CUDA_CHECK(cudaMalloc((void**)&scaleFactor, sizeof(float)*pd));
    CUDA_CHECK(cudaMemcpy(scaleFactor, scaleFactorHost, sizeof(float)*pd, cudaMemcpyHostToDevice));
    
    createMatrix<pd><<<blocks, blockSize>>>(w, h,
    					    features,
    					    table->table_entries,
    					    table->table_capacity,
    					    table->table_keys,
					    scaleFactor,
					    matrix);
    CUDA_POST_KERNEL_CHECK;

    // fix duplicate hash table entries
    int cleanBlockSize = 32;
    dim3 cleanBlocks((num_points-1)/cleanBlockSize+1, 2*(pd+1), 1);
    cleanHashTable<pd><<<cleanBlocks, cleanBlockSize>>>(2*num_points*(pd+1),
         table->table_entries, table->table_capacity, table->table_keys);
    CUDA_POST_KERNEL_CHECK;
    
    blocks.y *= pd+1;
    resetIndex<pd><<<blocks, blockSize>>>(w, h, matrix, table->table_entries) ;
    
    // Clean intermediate variables
    delete[] scaleFactorHost;
    CUDA_CHECK(cudaFree(scaleFactor));
}

template<int vd, int pd, typename Dtype>
void gpu_compute(Dtype* out, const Dtype* in, const HashTable* table,
  const MatrixEntry* matrix, 
  int w, int h, 
  bool reverse, bool add){

  // Create table_values
  int num_points = w*h ;
  float *table_values ;
  CUDA_CHECK(cudaMalloc((void**)&table_values, sizeof(float)*(vd+1)*num_points*(pd+1))) ;
  CUDA_CHECK(cudaMemset(table_values, 0, num_points*(vd+1)*(pd+1)*sizeof(float))) ;

  dim3 blocks((w-1)/8+1, (h-1)/8+1, 1);
  dim3 blockSize(8, 8, 1);
  
  // splat splits by color, so extend the y coordinate to our blocks to represent that
  blocks.y *= pd+1;
  splatCache<pd, vd><<<blocks, blockSize>>>(w, h, in, matrix,
   table->table_entries,
   table_values);
  //splat<pd, vd><<<blocks, blockSize>>>(w, h, values.device, matrix.device);
  CUDA_POST_KERNEL_CHECK;

  // blur
  int cleanBlockSize = 32;
  dim3 cleanBlocks((num_points-1)/cleanBlockSize+1, 2*(pd+1), 1);
  float *newValues;
  float *oldValues;
  size_t size =  num_points*(pd+1)*(vd+1)*sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&(newValues), size));
  CUDA_CHECK(cudaMalloc((void**)&(oldValues), size));
  CUDA_CHECK(cudaMemset(newValues, 0, size));
  for (int color = reverse?pd:0; color <= pd && color>=0; reverse?color--:color++) {
    blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(num_points*(pd+1), newValues,
     matrix,
     table->table_entries,
     table->table_keys,
     table->table_capacity,
     table_values,
     color);
    CUDA_POST_KERNEL_CHECK;
    swapHashTableValues(oldValues, newValues, table_values, size);
  }

  // slice
  blocks.y /= (pd+1);
  slice<pd, vd><<<blocks, blockSize>>>(w, h, out, matrix, table_values, add);
  CUDA_POST_KERNEL_CHECK;
  
  // Free memory
  CUDA_CHECK(cudaFree(table_values)) ;
  CUDA_CHECK(cudaFree(newValues)) ;
  CUDA_CHECK(cudaFree(oldValues)) ;
}

void ModifiedPermutohedral::init_gpu(const float* features, int num_dimensions, int w, int h) {
  //Initialize Hash table
  table.createHashTable(w*h*(num_dimensions+1), num_dimensions);
  CUDA_CHECK(cudaMalloc((void **)&matrix, sizeof(MatrixEntry)*(w*h*(num_dimensions+1))));
  w_ = w ;
  h_ = h ;
  d_ = num_dimensions ;
  N_ = w*h ;
  switch(num_dimensions){
    case 2:
      gpu_init<2>(features, &table, matrix,  w, h);
      break;
    case 5:
      gpu_init<5>(features, &table, matrix, w, h);
      break;
    default:
      LOG(FATAL) << "num_dimensions should be 2 or 5";
  } 
}

void ModifiedPermutohedral::compute_gpu(float* out, const float* in, int value_size, bool reverse, bool add) const {
  switch(1000*value_size+d_){
    case 1002: gpu_compute<1, 2, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    case 2002: gpu_compute<2, 2, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    case 3002: gpu_compute<3, 2, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    case 1005: gpu_compute<1, 5, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    case 2005: gpu_compute<2, 5, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    case 3005: gpu_compute<3, 5, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    case 21002: gpu_compute<21, 2, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    case 21005: gpu_compute<21, 5, float>(out, in, &table, matrix, w_, h_, reverse, add); break;
    default:
      LOG(FATAL) << "num_dimensions should be 1, 3 or 21";
  } 
}

void ModifiedPermutohedral::compute_gpu(double* out, const double* in, int value_size, bool reverse, bool add) const {
//TODO : view that later on 
  /*switch(1000*value_size+d_){
    case 1002: gpu_compute<1, 2, double>(out, in, table, matrix, w_, h_); break;
    case 2002: gpu_compute<2, 2, double>(out, in, table, matrix, w_, h_); break;
    case 3002: gpu_compute<3, 2, double>(out, in, table, matrix, w_, h_); break;
    case 1005: gpu_compute<1, 5, double>(out, in, table, matrix, w_, h_); break;
    case 2005: gpu_compute<2, 5, double>(out, in, table, matrix, w_, h_); break;
    case 3005: gpu_compute<3, 5, double>(out, in, table, matrix, w_, h_); break;
    default:
      LOG(FATAL) << "num_dimensions should be 1 or 3";
  } */
}

}//namespace caffe
