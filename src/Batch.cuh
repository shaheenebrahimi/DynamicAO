#pragma once

#include <vector>
#include "Matrix.cuh"

class Batch {
public:
	Batch();
	Batch(Shape dim, size_t batchSize);
	Batch(const std::vector<Matrix>& matrices);
	Batch(const std::vector<std::shared_ptr<float>> &matrices, Shape dim);
	~Batch();

	size_t batchSize; // how many matrices
	Shape matrixDim; // dimension of each matrix

	std::shared_ptr<float> data_device;
	std::shared_ptr<float> data_host;

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape matrixDim, size_t batchSize);

	void copyHostToDevice();
	void copyDeviceToHost();

	std::string to_string();
	
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();
};