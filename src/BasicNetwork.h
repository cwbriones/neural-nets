#ifndef BASICNETWORK_H
#define BASICNETWORK_H

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

// Type representing an entry of training data.
//
// The first vector is the input to the network and the 
// second vector is the desired output.
typedef std::pair<VectorXd, VectorXd> DataPair;
typedef std::vector<DataPair> Dataset;

// Implementation of a basic feed forward neural network
class BasicNetwork
{
public:
    BasicNetwork(const std::vector<size_t>& sizes);
    VectorXd feed_forward(const VectorXd& input);

    /*
     * Trains the neural network using the given training data set
     * over a given number of epochs using the method of stochastic
     * gradient descent with the backpropagation algorithm.
     *
     * Other Parameters:
     * eta - the learning rate
     * test_data - test data with which to show the progress of the training
     */
    void TrainSGD(const std::vector<DataPair>& training_data, 
        const size_t epochs, 
        const size_t mini_batch_size, 
        const float  eta, 
        const  std::vector<DataPair>& test_data);
    void evaluate(std::vector<DataPair>& test_data);
private:
    void update_mini_batch(std::vector<DataPair>& mini_batch, const float eta);
    DataPair back_propagation(const DataPair& training_example);
    VectorXd cost_derivative(const VectorXd& output_activations, const VectorXd& y);

    const std::vector<size_t> sizes_;
    const size_t num_layers_;

    std::vector<MatrixXd> weights_;
    std::vector<MatrixXd> biases_;
};

#endif
