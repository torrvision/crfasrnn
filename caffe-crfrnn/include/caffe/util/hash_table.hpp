#ifndef CAFFE_HASH_TABLE_HPP
#define CAFFE_HASH_TABLE_HPP

#include <cstdlib>
#include <vector>
#include <cassert>
#include <cstdio>
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"

// turn this on if you want to get slighly less memory consumption and slightly longer run times.
//#define LINEAR_D_MEMORY

/*
      Hash Table
*/
#define USE_CUSTOM_MODULO

namespace caffe{

class HashTable
{
  public:
    float *table_values;
    int *table_entries;
    unsigned int table_capacity;
    signed short *table_zeros;
    char *table_rank;
    signed short *table_keys;
    bool create;

    HashTable() : create(false) {}
    
    void createHashTable(const int capacity, const int kd, const int vd){
      // Initialize table_capacity
      // TODO? use symbol to go in constant memory instead
      CUDA_CHECK(cudaMalloc((void **) &table_capacity, 1));
      CUDA_CHECK(cudaMemCpy(table_capacity, &capacity, sizeof(unsigned int), CudaMemcpyHostToDevice));

      // Initialize table_values
      CUDA_CHECK(cudaMalloc((void **) &table_values, capacity*vd*sizeof(float)));
      CUDA_CHECK(cudaMemSet(table_values, 0, capacity*vd*sizeof(float)));
      
      // Initialize table_entries
      CUDA_CHECK(cudaMalloc((void **) &table_entries, 2*capacity*sizeof(int)));
      CUDA_CHECK(cudaMemSet(table_entries, -1, 2*capacity*sizeof(int)));
      
      #ifdef LINEAR_D_MEMORY
      // Initialize table_values
      //CUDA_CHECK(cudaMalloc((void **) &table_values, capacity*vd*sizeof(float)));
      //CUDA_CHECK(cudaMemSet(table_capacity, 0, capacity*vd*sizeof(unsigned int)));     
      
      // Initialize table_values
      //CUDA_CHECK(cudaMalloc((void **) &table_values, capacity*vd*sizeof(float)));
      //CUDA_CHECK(cudaMemSet(table_capacity, 0, capacity*vd*sizeof(unsigned int)));   
      
      #else
      // Initialize table_values
      CUDA_CHECK(cudaMalloc((void **) &table_keys, capacity*kd*sizeof(float)));
      CUDA_CHECK(cudaMemSet(table_keys, 0, capacity*kd*sizeof(float)));    
      #endif
      // Set create to true
      create = true;
    }
    
    void resetHashTable(){
      CUDA_CHECK(cudaMemset((void*)table_values, 0, table_capacity*vd*sizeof(float)));
    }
    
    ~HashTable(){
      if(create){
        // Free all pointers
        CUDA_CHECK(cudaFree(tale_values));
        CUDA_CHECK(cudaFree(tale_entries));
        CUDA_CHECK(cudaFree(tale_capacity));
        #ifdef LINEAR_D_MEMORY
        CUDA_CHECK(cudaFree(tale_zeros));
        CUDA_CHECK(cudaFree(tale_rank));
        #else
        CUDA_CHECK(cudaFree(tale_keys));
        #endif
        }
    }

}//namespace caffe
#endif //CAFFE_HASH_TABLE_HPP
