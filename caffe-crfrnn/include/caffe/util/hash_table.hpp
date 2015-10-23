#ifndef CAFFE_HASH_TABLE_HPP
#define CAFFE_HASH_TABLE_HPP

#include <cstdlib>
#include <vector>
#include <cassert>
#include <cstdio>
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"

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
    signed short *table_keys;
    bool create;

    HashTable() : create(false) {}
    
    void createHashTable(const int capacity, const int kd, const int vd){
      #ifndef CPU_ONLY
      // TODO? use symbol to go in constant memory instead
      // Initialize table_capacity
      table_capacity = (unsigned int)capacity ;

      // Initialize table_values
      CUDA_CHECK(cudaMalloc((void **) &table_values, capacity*vd*sizeof(float)));
      CUDA_CHECK(cudaMemset(table_values, 0, capacity*vd*sizeof(float)));
      
      // Initialize table_entries
      CUDA_CHECK(cudaMalloc((void **) &table_entries, 2*capacity*sizeof(int)));
      CUDA_CHECK(cudaMemset(table_entries, -1, 2*capacity*sizeof(int)));
      
      // Initialize table_values
      CUDA_CHECK(cudaMalloc((void **) &table_keys, capacity*kd*sizeof(float)));
      CUDA_CHECK(cudaMemset(table_keys, 0, capacity*kd*sizeof(float)));    

      // Set create to true
      create = true;
      #endif // CPU_ONLY
    }
    
    void resetHashTable(int vd){
      #ifndef CPU_ONLY
      CUDA_CHECK(cudaMemset((void*)table_values, 0, table_capacity*vd*sizeof(float)));
      #endif //CPU_ONLY
    }
    
    ~HashTable(){
      #ifndef CPU_ONLY
      if(create){
        // Free all pointers
        CUDA_CHECK(cudaFree(table_values));
        CUDA_CHECK(cudaFree(table_entries));
        CUDA_CHECK(cudaFree(table_keys));
        }
      #endif //CPU_ONLY
    }

};
}//namespace caffe
#endif //CAFFE_HASH_TABLE_HPP
