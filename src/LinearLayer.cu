#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <sstream>


#include "LinearLayer.cuh"
#include "NNException.h"

//__global__ void linearLayerForward( float* W, float* A, float* Z, float* b,
//									int W_x_dim, int W_y_dim,
//									int A_x_dim, int A_y_dim) {
//
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//	// W is treated as transposed
//	int Z_x_dim = A_x_dim;
//	int Z_y_dim = W_y_dim;
//
//	float Z_value = 0;
//
//	if (row < Z_y_dim && col < Z_x_dim) {
//		for (int i = 0; i < W_x_dim; i++) {
//			Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
//		}
//		Z[row * Z_x_dim + col] = Z_value + b[row];
//	}
//}

__global__ void linearLayerForward(float* A, float* W, float* b, float* Z, int N, int P, int Q) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < P && col < Q) {
		float z = 0.0;
		for (int k = 0; k < N; ++k) {
			z += A[row * N + k] * W[k * Q + col];
		}
		Z[row * Q + col] = z + b[col];
	}
}

__global__ void linearLayerForwardBatch(float* bA, float* W, float* b, float* bZ, int N, int P, int Q, int batchSize) {

	int batch = blockIdx.z * blockDim.z + threadIdx.z;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (batch < batchSize && row < P && col < Q) {
		float z = 0.0;
		for (int i = 0; i < N; ++i) {
			z += bA[batch * P * N + row * N + i] * W[i * Q + col];
		}
		bZ[batch * P * Q + row * Q + col] = z + b[col];
	}
}

__global__ void linearLayerBackprop(float* W, float* dZ, float *dA,
									int W_x_dim, int W_y_dim,
									int dZ_x_dim, int dZ_y_dim) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// W is treated as transposed
	int dA_x_dim = dZ_x_dim;
	int dA_y_dim = W_x_dim;

	float dA_value = 0.0f;

	if (row < dA_y_dim && col < dA_x_dim) {
		for (int i = 0; i < W_y_dim; i++) {
			dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
		}
		dA[row * dA_x_dim + col] = dA_value;
	}
}


//__global__ void linearLayerUpdateWeights(  float* dZ, float* A, float* W,
//										   int dZ_x_dim, int dZ_y_dim,
//										   int A_x_dim, int A_y_dim,
//										   float learning_rate) {
//
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//
//	// A is treated as transposed
//	int W_x_dim = A_y_dim;
//	int W_y_dim = dZ_y_dim;
//
//	float dW_value = 0.0f;
//
//	if (row < W_y_dim && col < W_x_dim) {
//		for (int i = 0; i < dZ_x_dim; i++) {
//			dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
//		}
//		W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
//	}
//}

//__global__ void linearLayerUpdateBias(  float* dZ, float* b,
//										int dZ_x_dim, int dZ_y_dim,
//										int b_x_dim,
//										float learning_rate) {
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (index < dZ_x_dim * dZ_y_dim) {
//		int dZ_x = index % dZ_x_dim;
//		int dZ_y = index / dZ_x_dim;
//		atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
//	}
//}

LinearLayer::LinearLayer(std::string name, Shape W_shape) :
	W(W_shape), b(1, W_shape.y)
{
	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

LinearLayer::~LinearLayer()
{ }

void LinearLayer::loadLayer(const std::string& stream) {
	std::stringstream ss(stream);
    float* weightBuf = new float[W.shape.x * W.shape.y];
    for (int i = 0; i < W.shape.x * W.shape.y; ++i) {
        ss >> weightBuf[i];
    }
    float* biasBuf = new float[b.shape.x * b.shape.y];
    for (int i = 0; i < b.shape.x * b.shape.y; ++i) {
        ss >> biasBuf[i];
    }
    W.setBuf(weightBuf);
    b.setBuf(biasBuf);
    delete[] weightBuf;
    delete[] biasBuf;
}

void LinearLayer::initializeWeightsRandomly() {
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copyHostToDevice();
}

void LinearLayer::initializeBiasWithZeros() {
	for (int x = 0; x < b.shape.x; x++) {
		b[x] = 0;
	}

	b.copyHostToDevice();
}

Matrix& LinearLayer::forward(Matrix& A) {
	assert(A.shape.y == W.shape.x);
	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);
	computeAndStoreLayerOutput(A);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");
	return Z;
}

Batch& LinearLayer::forwardBatch(Batch& batchedA) {
	assert(batchedA.matrixDim.y == W.shape.x);
	this->batchedA = batchedA;
	Shape Z_shape(A.shape.x, W.shape.y);
	batchedZ.allocateMemoryIfNotAllocated(Z_shape, batchedA.batchSize);
	computeAndStoreLayerBatchedOutput(batchedA);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");
	return batchedZ;
}

void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
	int N = W.shape.x, P = A.shape.x, Q = W.shape.y; // (P, N) * (N, Q)
	dim3 block_size(32, 32);
	dim3 num_of_blocks(ceilf(P / (float)block_size.x), ceilf(Q / (float)block_size.y));
	linearLayerForward <<<num_of_blocks, block_size>>> (A.data_device.get(), W.data_device.get(), b.data_device.get(), Z.data_device.get(), N, P, Q);
}

void LinearLayer::computeAndStoreLayerBatchedOutput(Batch& batchedA) {
	int N = W.shape.x, P = A.shape.x, Q = W.shape.y, bs = batchedA.batchSize; // (P, N) * (N, Q)
	dim3 block_size(16, 16, 4);
	dim3 num_of_blocks(ceilf(P / (float)block_size.x), ceilf(Q / (float)block_size.y), ceilf(bs / (float)block_size.z));
	linearLayerForwardBatch <<<num_of_blocks, block_size>>> (batchedA.data_device.get(), W.data_device.get(), b.data_device.get(), batchedZ.data_device.get(), N, P, Q, bs);
}

//Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
//	dA.allocateMemoryIfNotAllocated(A.shape);
//
//	//computeAndStoreBackpropError(dZ);
//	NNException::throwIfDeviceErrorsOccurred("Cannot perform back propagation.");
//
//	updateBias(dZ, learning_rate);
//	NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");
//
//	updateWeights(dZ, learning_rate);
//	NNException::throwIfDeviceErrorsOccurred("Cannot perform weights update.");
//
//	return dA;
//}

//void LinearLayer::computeAndStoreBackpropError(Matrix& dZ) {
//	dim3 block_size(8, 8);
//	dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
//						(A.shape.y + block_size.y - 1) / block_size.y);
//	linearLayerBackprop<<<num_of_blocks, block_size>>>( W.data_device.get(),
//														dZ.data_device.get(),
//														dA.data_device.get(),
//														W.shape.x, W.shape.y,
//														dZ.shape.x, dZ.shape.y);
//}

//void LinearLayer::updateWeights(Matrix& dZ, float learning_rate) {
//	dim3 block_size(8, 8);
//	dim3 num_of_blocks(	(W.shape.x + block_size.x - 1) / block_size.x,
//						(W.shape.y + block_size.y - 1) / block_size.y);
//	linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
//															A.data_device.get(),
//															W.data_device.get(),
//															dZ.shape.x, dZ.shape.y,
//															A.shape.x, A.shape.y,
//															learning_rate);
//}
//
//void LinearLayer::updateBias(Matrix& dZ, float learning_rate) {
//	dim3 block_size(256);
//	dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
//	linearLayerUpdateBias<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
//														 b.data_device.get(),
//														 dZ.shape.x, dZ.shape.y,
//														 b.shape.x, learning_rate);
//}

int LinearLayer::getXDim() const {
	return W.shape.x;
}

int LinearLayer::getYDim() const {
	return W.shape.y;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}
