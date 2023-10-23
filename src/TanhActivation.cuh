#pragma once

#include "NNLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

class TanhActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	TanhActivation(std::string name);
	~TanhActivation();

	Matrix& forward(Matrix& Z);
	//Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};