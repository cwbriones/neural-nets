#ifndef BASICNETWORK_H
#define BASICNETWORK_H

#include <Eigen/Dense>
#include <vector>
#include "Util.h"

using namespace Eigen;

// Implementation of a basic feed forward neural network
class BasicNetwork
{
public:
    BasicNetwork(const std::vector<size_t>& sizes);

    /*
     * Trains the neural network using the given training data set
     * over a given number of epochs using the method of stochastic
     * gradient descent with the backpropagation algorithm.
     *
     * Other Parameters:
     * eta - the learning rate
     * test_data - test data with which to show the progress of the training
     */
    void TrainSGD(std::vector<DataPair> training_data, 
        const size_t epochs, 
        const size_t mini_batch_size, 
        const float  eta, 
        const  std::vector<DataPair>& test_data);
    size_t evaluate(std::vector<DataPair>& test_data);

    VectorXd feed_forward(VectorXd a);

    static double sigmoid_func(double z);
    static VectorXd sigmoid_vec(const VectorXd& z);
    static VectorXd sigmoid_prime_vec(const VectorXd& z);
private:
    void update_mini_batch(const std::vector<DataPair>& mini_batch, const float eta);

    /*
     * Computes the gradient using the back propagation algorithm.
     * Returns:
     *  A pair with first = nabla_w and second = nabla_b
     */
    std::pair<MatrixList, MatrixList> back_propagation(const DataPair& training_example);
    VectorXd cost_derivative(const VectorXd& output_activations, const VectorXd& y);

    const std::vector<size_t> sizes_;
    const size_t num_layers_;

    MatrixList weights_;
    MatrixList biases_;
};

#endif
