#include "TanhActivation.cuh"
#include "NNException.h"
#include <iostream>

//__device__ float tanhActivation(float x) {
//	return 1.0f / (1 + exp(-x));
//}

__global__ void tanhActivationForward(float* Z, float* A,
										 int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = tanh(Z[index]);
	}
}

//__global__ void tanhActivationBackprop(float* Z, float* dA, float* dZ,
//										  int Z_x_dim, int Z_y_dim) {
//
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (index < Z_x_dim * Z_y_dim) {
//		dZ[index] = dA[index] * tanh(Z[index]) * (1 - tanh(Z[index]));
//	}
//}

TanhActivation::TanhActivation(std::string name) {
	this->name = name;
}

TanhActivation::~TanhActivation()
{ }

Matrix& TanhActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	tanhActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(), Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform tanh forward propagation.");

	return A;
}

//Matrix& TanhActivation::backprop(Matrix& dA, float learning_rate) {
//	dZ.allocateMemoryIfNotAllocated(Z.shape);
//
//	dim3 block_size(256);
//	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
//	tanhActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
//															 dZ.data_device.get(),
//															 Z.shape.x, Z.shape.y);
//	NNException::throwIfDeviceErrorsOccurred("Cannot perform tanh back propagation");
//
//	return dZ;
//}