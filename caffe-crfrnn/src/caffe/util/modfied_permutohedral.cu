#define BLOCK_SIZE 64

#define _DEBUG
#include <stdio.h>
#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/hash_helper.cu"

//TODO include w and h in Modified permutohedral def

namespace caffe {

__global__ static void createMatrix(const int num_points, const int pd
				    const float *positions,
				    const float *scaleFactor,
				    MatrixEntry *matrix) {

    // scanline order
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool outOfBounds = (idx>=num_points) ;

    // 8x8 blocks
    //const int x = threadIdx.x + blockIdx.x * blockDim.x;
    //const int y = threadIdx.y + blockIdx.y * blockDim.y;
    //const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
    //const int idx = y*w + x;
    //const bool outOfBounds = (x >= w) || (y >= h);
  
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

        #ifdef LINEAR_D_MEMORY
	for (int i = 0; i <= pd; i++) {
	    table_zeros[idx*(pd+1)+i] = myGreedy[i];
	    table_rank[idx*(pd+1)+i] = myRank[i];
	}
	#endif

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
    unsigned int cumulative_hash = hash(myGreedy, pd);
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
	    r.index = hashTableInsert<pd>(cumulative_hash, myKey, idx*(pd+1)+color);
	    #else
	    r.index = hashTableInsert<pd>(myKey, idx*(pd+1)+color);
	    #endif
	    r.weight = myBarycentric[color];
	    matrix[idx*(pd+1) + color] = r;
	}
    }
}


__global__ static void cleanHashTable(int n, int kd, MatrixEntry *matrix) {
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

        #ifdef LINEAR_D_MEMORY
        // Get my key
        short myKey[kd];
        generateKey<kd>(*e, myKey);
	*e = hashTableRetrieve<kd>(myKey);
        #else
	*e = hashTableRetrieve<kd>(table_keys + *e*kd);
	#endif
    }
}

void ModifiedPermutohedral::init_gpu(const float* features, int num_dimensions, int num_points) {

    unsigned int blocks = (num_points-1)/64 + 1;
    unsigned int blockSize = 64;
    float blurVariance = 0.5 ;
    float * scaleFactor;
    float* scaleFactorHost = new float[num_dimensions];
    
    // Create Scale factor vector and give it to GPU
    // num_dimensions is likely to be low so do that 
    // on the CPU
    for (int i = 0; i < num_dimensions; i++) {
	scaleFactorHost[i] = (num_dimensions+1)*sqrtf((1.0/6 + blurVariance)/((i+1)*(i+2)));
    }
    CUDA_CHECK(cudaMalloc((void**)&scaleFactor, sizeof(float)*num_dimensions));
    CUDA_CHECK(cudaMemCpy(scaleFactor, scaleFactorHost, sizeof(float)*num_dimensions, cudaMemcpyHostToDevice));
    
    // Allocate matrix and create hash table
    CUDA_CHECK(cudaMalloc((void **)&matrix, sizeof(Matrix)*(num_points*(num_dimensions+1))));
    table.createHashTable(num_points*(num_dimensions+1), num_dimensions, 3);

    // Populate memory for hash helpers
    unsigned long long int __host_two32 = ((unsigned long long int)1)<<32;
    unsigned int __host_div_c = 2*(num_points*(num_dimensions+1));
    unsigned int __host_div_l = ceilf(logf((float)__host_div_c) / logf(2.0f));
    unsigned int __host_div_m = (__host_two32<<__host_div_l)/__host_div_c - __host_two32 + 1;
    /*CUDA_CHECK(cudaMemcpy((char*)&__div_c, &__host_div_c, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy((char*)&__div_l, &__host_div_l, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy((char*)&__div_m, &__host_div_m, sizeof(unsigned int)));

    // Populate constant memory with hash of offset vectors
    unsigned int hOffset_host[num_dimensions+1];
    signed short offset[num_dimensions+1];
    for (int i = 0; i < num_dimensions; offset[i] = 1, i++);
    for (int i = 0; i <= num_dimensions; i++) {
      offset[i] -= num_dimensions+1; hOffset_host[i] = hash<num_dimensions>(offset); offset[i] += num_dimensions+1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol((char*)&hOffset, &hOffset_host, sizeof(unsigned int)*(num_dimensions+1)));
*/
//TODO hash and hashtable insert
    createMatrix<num_dimensions><<<blocks, blockSize>>>(num_dimensions, features,
					    scaleFactor,
					    matrix);
    CUDA_POST_KERNEL_CHECK;

    // fix duplicate hash table entries
    int cleanBlockSize = 32;
    dim3 cleanBlocks((num_points-1)/cleanBlockSize+1, 2*(num_dimensions+1), 1);
    cleanHashTable<<<cleanBlocks, cleanBlockSize>>>(2*num_points*(num_dimensions+1), num_dimensions, matrix);
    CUDA_POST_KERNEL_CHECK;
    
    // Clean intermediate variables
    // TODO : see what can be further cleaned
    delete[] scaleFactorHost;
    CUDA_CHECK(cudaFree(scaleFactor));
}

}//namespace caffe
