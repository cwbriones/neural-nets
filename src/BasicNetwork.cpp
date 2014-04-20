#include "BasicNetwork.h"

void BasicNetwork::TrainSGD(const std::vector<DataPair>& training_data,
    const size_t epochs,
    const size_t mini_batch_size,
    const float  eta,
    const  std::vector<DataPair>& test_data) {
}

void BasicNetwork::evaluate(std::vector<DataPair>& test_data) {
    std::functiona:q
}

void BasicNetwork::update_mini_batch(std::vector<DataPair>& mini_batch,
    const float eta) {
}

DataPair BasicNetwork::back_propagation(const DataPair& training_example) {
}

VectorXd BasicNetwork::cost_derivative(const VectorXd& output_activations,
    const VectorXd& y) {
}
