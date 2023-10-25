#pragma once
#include "NNLayer.h"
#include "Batch.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// for unit testing purposes only
namespace {
	class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	class LinearLayerTest_ShouldReturnDerivativeAfterBackprop_Test;
	class LinearLayerTest_ShouldUptadeItsBiasDuringBackprop_Test;
	class LinearLayerTest_ShouldUptadeItsWeightsDuringBackprop_Test;
}

class LinearLayer : public NNLayer {
public:
	LinearLayer(std::string name, Shape W_shape); // transposed W
	~LinearLayer();

	void loadLayer(const std::string& stream);

	Matrix& forward(Matrix& A);
	Batch& forwardBatch(Batch& batchedA);

	//Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix A;
	Matrix dA;

	Batch batchedZ;
	Batch batchedA;
	//Batch batcheddA;

	void initializeBiasWithZeros();
	void initializeWeightsRandomly();

	//void computeAndStoreBackpropError(Matrix& dZ);

	void computeAndStoreLayerOutput(Matrix& A);
	void computeAndStoreLayerBatchedOutput(Batch& batchedA);

	//void updateWeights(Matrix& dZ, float learning_rate);
	//void updateBias(Matrix& dZ, float learning_rate);
};
