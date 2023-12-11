#include "Evaluator.cuh"
#include "NNLayer.h"
#include "NNException.h"
#include "LinearLayer.cuh"
#include "TanhActivation.cuh"

#include <string>
#include <sstream>
#include <fstream>

using namespace std;

Evaluator::Evaluator(float learning_rate) :
	learning_rate(learning_rate)
{ }

Evaluator::Evaluator(const std::string &model) :
	learning_rate(0.01)
{
	loadEvaluator(model);
}

Evaluator::~Evaluator() {
	for (auto layer : layers) {
		delete layer;
	}
}

void Evaluator::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix Evaluator::forward(Matrix X) {
	Matrix Z = X;
	for (auto layer : layers) {
		Z = layer->forward(Z);
	}
	Y = Z;
	return Y;
}

Batch Evaluator::forwardBatch(Batch batchedX) {
	Batch batchedZ = batchedX;
	for (auto layer : layers) {
		batchedZ = layer->forwardBatch(batchedZ);
	}
	batchedY = batchedZ;
	return batchedY;
}

void Evaluator::loadEvaluator(const std::string& model) {
	ifstream in;
	in.open(model);
	if (!in.good()) {
	    cout << "Cannot read " << model << endl;
	    return;
	}
	//cout << "Loading " << model << endl;
	
	string line;
	stringstream ss;
	
	// Get meta data
	getline(in, line);
	ss = stringstream(line);
	int numLayers;
	ss >> numLayers;
	
	//cout << "layers: " << numLayers << endl;
	
	// Layer data
	for (int l = 0; l < numLayers; ++l) {
	    getline(in, line);
	    ss = stringstream(line);
	    // Get layer data
	    int inputs, outputs;
	    ss >> inputs; ss >> outputs;
		LinearLayer* layer = new LinearLayer("layer"+to_string(l), Shape(inputs, outputs));
	    //cout << "inputs: " << inputs << " outputs: " << outputs << endl;
	    // Get values
	    getline(in, line);
	    ss = stringstream(line);
	    layer->loadLayer(line);
	    layers.push_back(layer);
		if (l != numLayers - 1) layers.push_back(new TanhActivation("tanh" + to_string(l)));
	}
	in.close();
}

std::string Evaluator::getInfo() {
	std::string info = "";
	for (auto layer : layers) {
		info += layer->getName() + "\n";
	}
	return info;
}

float Evaluator::evaluate(const Matrix &input) {
	Matrix output = forward(input);
	output.copyDeviceToHost();
	return output[0];
}

std::vector<float> Evaluator::evaluateBatch(const Batch& input) {
	std::vector<float> res(input.batchSize);
	Batch output = forwardBatch(input);
	output.copyDeviceToHost();
	for (int b = 0; b < input.batchSize; ++b) {
		res[b] = output.data_host.get()[b];
	}
	return res;
}

void Evaluator::sharedBatchCompute(const Batch& input, struct cudaGraphicsResource** outputResource)
{	
	float* output;
	cudaGraphicsMapResources(1, outputResource, 0);

	size_t numBytes;
	cudaGraphicsResourceGetMappedPointer((void**)&output, &numBytes, *outputResource);


	Batch res = forwardBatch(input);

	cudaMemcpy(output, res.data_device.get(), numBytes, cudaMemcpyDeviceToDevice);

	// unmap buffer object
	cudaGraphicsUnmapResources(1, outputResource, 0);
}

//void Evaluator::backprop(Matrix predictions, Matrix target) {
//	dY.allocateMemoryIfNotAllocated(predictions.shape);
//	Matrix error = bce_cost.dCost(predictions, target, dY);
//
//	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
//		error = (*it)->backprop(error, learning_rate);
//	}
//
//	cudaDeviceSynchronize();
//}

std::vector<NNLayer*> Evaluator::getLayers() const {
	return layers;
}
