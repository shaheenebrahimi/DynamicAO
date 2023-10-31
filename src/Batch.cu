#include "Batch.cuh"
#include "NNException.h"

Batch::Batch() {
	this->device_allocated = false;
	this->host_allocated = false;
	this->matrixDim = Shape(1,1);
	this->batchSize = 1;
}

Batch::Batch(Shape dim, size_t batchSize) {
	this->device_allocated = false;
	this->host_allocated = false;
	this->matrixDim = dim;
	this->batchSize = batchSize;
	allocateMemory();
}

Batch::Batch(const std::vector<Matrix>& matrices) {
	this->device_allocated = false;
	this->host_allocated = false;
	this->matrixDim = matrices[0].shape;
	this->batchSize = matrices.size();
	allocateMemory();
	for (int batch = 0; batch < batchSize; ++batch) {
		memcpy(data_host.get() + batch * matrixDim.x * matrixDim.y, matrices[batch].data_host.get(), matrixDim.x * matrixDim.y * sizeof(float));
	}
	copyHostToDevice();
}

Batch::Batch(Shape dim, const std::vector<std::vector<float>> &matrices) {
	this->device_allocated = false;
	this->host_allocated = false;
	this->batchSize = matrices.size();
	this->matrixDim = dim;
	allocateMemory();
	for (int batch = 0; batch < batchSize; ++batch) {
		memcpy(data_host.get() + batch * matrixDim.x * matrixDim.y, matrices[batch].data(), matrixDim.x * matrixDim.y * sizeof(float));
	}
	copyHostToDevice();
}

Batch::~Batch() {

}

void Batch::allocateCudaMemory() {
	if (!device_allocated) {
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, batchSize * matrixDim.x * matrixDim.y * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		data_device = std::shared_ptr<float>(device_memory,
			[&](float* ptr) { cudaFree(ptr); });
		device_allocated = true;
	}
}

void Batch::allocateHostMemory() {
	if (!host_allocated) {
		data_host = std::shared_ptr<float>(new float[batchSize * matrixDim.x * matrixDim.y],
			[&](float* ptr) { delete[] ptr; });
		host_allocated = true;
	}
}

void Batch::allocateMemory() {
	allocateCudaMemory();
	allocateHostMemory();
}

void Batch::allocateMemoryIfNotAllocated(Shape matrixDim, size_t batchSize) {
	if (!device_allocated && !host_allocated) {
		this->matrixDim = matrixDim;
		this->batchSize = batchSize;
		allocateMemory();
	}
}

void Batch::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device.get(), data_host.get(), batchSize * matrixDim.x * matrixDim.y * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Batch::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host.get(), data_device.get(), batchSize * matrixDim.x * matrixDim.y * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}
std::string Batch::to_string() {
	std::string out = "";
	for (int b = 0; b < batchSize; ++b) {
		for (int i = 0; i < matrixDim.x * matrixDim.y; ++i) {
			out += std::to_string(data_host.get()[b * matrixDim.x * matrixDim.y + i]) + " ";
		}
		out += "\n";
	}
	return out;
}
