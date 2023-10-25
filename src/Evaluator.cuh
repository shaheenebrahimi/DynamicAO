#pragma once

#include <vector>
#include <string>
#include <memory>
#include "NNLayer.h"
#include "Matrix.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
//#include "nn_utils/bce_cost.hh"

class Evaluator {
public:
	Evaluator(float learning_rate = 0.01);
	~Evaluator();

	void loadEvaluator(const std::string& model);
	float evaluate(const Matrix &input);
	std::vector<float> evaluateBatch(const Batch& input);

	Matrix forward(Matrix X);
	Batch forwardBatch(Batch batchedX);
	//void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::string getInfo();
	std::vector<NNLayer*> getLayers() const;

private:
	std::vector<NNLayer*> layers;
	//BCECost bce_cost;

	Matrix Y;
	Matrix dY;
	Batch batchedY;

	float learning_rate;
};
