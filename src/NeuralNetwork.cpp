#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork() {

}

void NeuralNetwork::loadNetwork(const std::string &filename) {
    ifstream in;
    in.open(filename);
    if(!in.good()) {
        cout << "Cannot read " << filename << endl;
        return;
    }
    cout << "Loading " << filename << endl;

    string line;
    stringstream ss;
    
    // Get meta data
    getline(in, line);
    ss = stringstream(line);
    ss >> numLayers;

    cout << "layers: " << numLayers << endl;

    // Layer data
    network.resize(numLayers);
    for (int l = 0; l < numLayers; ++l) {
        getline(in, line);
        ss = stringstream(line);

        // Get connection data
        int inputs, outputs;
        ss >> inputs; ss >> outputs;
        shared_ptr<Connection> con = make_shared<Connection>(inputs, outputs);
        cout << "inputs: " << inputs << " outputs: " << outputs << endl;
        // Get values
        getline(in, line);
        ss = stringstream(line);
        for (int i = 0; i < inputs; ++i) { // populate weights
            for (int o = 0; o < outputs; ++o) {
                float val;
                ss >> val;
                con->setWeight(i, o, val);
            }
        }
        for (int o = 0; o < outputs; ++o) { // populate biases
            float val;
            ss >> val;
            con->setBias(o, val);
        }
        network[l] = con;
    }
    in.close();
}

void NeuralNetwork::activation(Eigen::MatrixXf &z) { // tanh
    for (int i = 0; i < z.rows(); ++i) {
        for (int j = 0; j < z.cols(); ++j) {
            z(i,j) = tanh(z(i,j)); // apply
        }
    }
}

float NeuralNetwork::evaluate(float u, float v) {
    Eigen::MatrixXf input (1, 2); input << u, v;
    Eigen::MatrixXf output = input;
    for (int i = 0; i < network.size(); ++i) {
        Eigen::MatrixXf z = output * network[i]->weights + network[i]->biases;
        if (i != network.size() - 1) activation(z);
        output = z; // move to next layer
    }
    return output(0);
}
