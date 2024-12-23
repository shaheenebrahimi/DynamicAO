#include "Matrix.cuh"
#include "NNException.h"


Matrix::Matrix(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape) :
	Matrix(shape.x, shape.y)
{ }

Matrix::Matrix(std::vector<float> data, Shape shape) : shape(shape), data_device(nullptr), data_host(nullptr), 
	device_allocated(false), host_allocated(false)
{
	allocateMemory();
	memcpy(data_host.get(), data.data(), shape.x * shape.y * sizeof(float));
	copyHostToDevice();
}

void Matrix::allocateCudaMemory() {
	if (!device_allocated) {
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		data_device = std::shared_ptr<float>(device_memory,
											 [&](float* ptr){ cudaFree(ptr); });
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
										   [&](float* ptr){ delete[] ptr; });
		host_allocated = true;
	}
}

void Matrix::allocateMemory() {
	allocateCudaMemory();
	allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		allocateMemory();
	}
}

void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

void Matrix::setBuf(float *buf) {
	allocateMemoryIfNotAllocated(Shape(shape.x, shape.y));
    memcpy(data_host.get(), buf, shape.x * shape.y * sizeof(float));
    copyHostToDevice();
}

std::string Matrix::to_string() {
	std::string out = "";
	int len = shape.x * shape.y;
	for (int i = 0; i < len; ++i) {
		out += std::to_string(data_host.get()[i]) + " ";
	}
	return out + '\n';
}


float& Matrix::operator[](const int index) {
	return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const {
	return data_host.get()[index];
}
