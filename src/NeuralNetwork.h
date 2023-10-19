#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <cmath>
#include <string>

struct Connection {
public:
    Connection(int inputs, int outputs) {
        this->inputs = inputs;
        this->outputs = outputs;
        weights.resize(inputs, outputs);
        biases.resize(1, outputs);
    }

    void setWeight(int x, int y, float value) {
        weights(x, y) = value;
    }

    void setBias(int x, float value) {
        biases(0, x) = value;
    }

    int inputs;
    int outputs;
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;
};

class NeuralNetwork {
public:
    NeuralNetwork();
    void loadNetwork(const std::string &filepath);
    void activation(Eigen::MatrixXf &z);
    float evaluate(float u, float v);
private:
    int numLayers;
    std::vector<std::shared_ptr<Connection>> network;

};

#endif