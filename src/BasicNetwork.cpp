#include <iostream>
#include <random>
#include "BasicNetwork.h"

BasicNetwork::BasicNetwork(const std::vector<size_t>& sizes) : 
    sizes_(sizes),
    num_layers_(sizes.size())
{
    weights_.reserve(num_layers_ - 1);
    biases_.reserve(num_layers_ - 1);

    for (int i = 0; i < sizes_.size() - 1; ++i) {
        const size_t x = sizes_[i];
        const size_t y = sizes_[i + 1];
        
        // Initialize the weights and biases
        // to normally distributed values

        auto rand_normal = [](double val){
            static std::random_device rd;
            static std::mt19937_64 gen(rd());
            static std::normal_distribution<> d(0, 1);

            return d(gen);
        };
        MatrixXd layer_weights = MatrixXd::Zero(x, y).unaryExpr(rand_normal);
        MatrixXd layer_biases = MatrixXd::Zero(x, 1).unaryExpr(rand_normal);

        weights_.push_back(layer_weights);
        biases_.push_back(layer_biases);
    }
}

void BasicNetwork::TrainSGD(const std::vector<DataPair>& training_data,
    const size_t epochs,
    const size_t mini_batch_size,
    const float  eta,
    const  std::vector<DataPair>& test_data) {

}

void BasicNetwork::evaluate(std::vector<DataPair>& test_data) {
}

void BasicNetwork::update_mini_batch(std::vector<DataPair>& mini_batch,
    const float eta) {
}

DataPair BasicNetwork::back_propagation(const DataPair& training_example) {
    return training_example;
}

VectorXd BasicNetwork::cost_derivative(const VectorXd& output_activations,
    const VectorXd& y) {

    return output_activations - y;
}
