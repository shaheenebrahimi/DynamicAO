#pragma once

#include <iostream>

#include "Matrix.cuh"
#include "Batch.cuh"

class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual Matrix& forward(Matrix& A) = 0;
	virtual Batch& forwardBatch(Batch& batchedA) = 0;

	//virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;

	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}
