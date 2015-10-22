#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/hash_table.hpp"

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/
namespace caffe {

class ModifiedPermutohedral
{
protected:
	struct Neighbors{
		int n1, n2;
		Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
		}
	};
	typedef struct MatrixEntry {
        int index;
        float weight;
      } Matrix;
	std::vector<int> offset_, rank_;
	std::vector<float> barycentric_;
	std::vector<Neighbors> blur_neighbors_;
	
	// GPU specific
      Matrix *matrix;
      HashTable table;

	
	// Number of elements, size of sparse discretized space, dimension of features
	int N_, M_, d_;

      void init_cpu(const float* features, int num_dimensions, int num_points);
      void init_gpu(const float* features, int num_dimensions, int num_points);

	void sseCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
      void sseCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

	void seqCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
	void seqCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

public:
	ModifiedPermutohedral();
	void init (const float* features, int num_dimensions, int num_points){
	  init_cpu(features, num_dimensions, num_points); 
	}
	void compute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
	void compute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;
};
}//namespace caffe
#endif //CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
