#pragma once

#include <memory>
#include <string>
#include <vector>

struct Shape {
	size_t x, y;
	Shape(size_t x = 1, size_t y = 1) {
		this->x = x; this->y = y;
	}
};

class Matrix {
public:
	Shape shape;

	std::shared_ptr<float> data_device;
	std::shared_ptr<float> data_host;

	Matrix(size_t x_dim = 1, size_t y_dim = 1);
	Matrix(std::vector<float> data, Shape shape);
	Matrix(Shape shape);

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();

	void setBuf(float* buf);
	std::string to_string();

	float& operator[](const int index);
	const float& operator[](const int index) const;

private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();
};
