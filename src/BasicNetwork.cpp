#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

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
        MatrixXd layer_weights = MatrixXd::Zero(y, x).unaryExpr(rand_normal);
        VectorXd layer_biases = VectorXd::Zero(y).unaryExpr(rand_normal);

        weights_.push_back(layer_weights);
        biases_.push_back(layer_biases);
    }
}

void BasicNetwork::TrainSGD(std::vector<DataPair> training_data,
    const size_t epochs,
    const size_t mini_batch_size,
    const float  eta,
    const  std::vector<DataPair>& test_data = {}) {

    int n_data = training_data.size();

    for (int i = 0; i < epochs; ++i) {
        std::cout << "Begin Epoch " << i + 1 << std::endl;

        std::cout << "Creating mini batches..." << std::endl;
        std::random_shuffle(training_data.begin(), training_data.end());
        std::vector<std::vector<DataPair>> mini_batches;

        // Create the mini batches
        for (int j = 0; j < n_data; j += mini_batch_size) {
            std::vector<DataPair> batch;
            for (int k = i; k < i + mini_batch_size; ++k) {
                batch.push_back(training_data[k]);
            }
            mini_batches.push_back(std::move(batch));
        }

        std::cout << "Updating..." << std::endl;
        for (auto& mini_batch : mini_batches) {
            update_mini_batch(mini_batch, eta);
        }
    }
}

void BasicNetwork::update_mini_batch(std::vector<DataPair>& mini_batch,
    const float eta) {

    // Initialize to matrices with 0s
    MatrixList nabla_b(biases_);
    MatrixList nabla_w(weights_);

    std::for_each(nabla_b.begin(), nabla_b.end(), [](MatrixXd& mat){mat.setZero();});
    std::for_each(nabla_w.begin(), nabla_w.end(), [](MatrixXd& mat){mat.setZero();});

    // Compute the gradient by applying the back-propagation
    // algorithm to each example in the mini batch and
    // summing the resulting lists of matrices elementwise.
    for (auto& example : mini_batch) {
        auto delta_nabla = back_propagation(example);

        auto delta_nabla_b = delta_nabla.first;
        auto delta_nabla_w = delta_nabla.second;

        for (int i = 0; i < num_layers_ - 1; ++i) {
            nabla_b[i] += delta_nabla_b[i];
            nabla_w[i] += delta_nabla_w[i];
        }
    }

    // Update the weights and biases based on the learning rate
    for (int i = 0; i < num_layers_ - 1; ++i) {
        biases_[i]  -= eta * nabla_b[i];
        weights_[i] -= eta * nabla_w[i];
    }
}

std::pair<MatrixList, MatrixList> BasicNetwork::back_propagation(const DataPair& training_example) {
    // Create empty matrix/vector for the gradient
    auto nabla_b(biases_);
    auto nabla_w(weights_);

    std::for_each(nabla_w.begin(), nabla_w.end(), [](MatrixXd& mat){mat.setZero();});
    std::for_each(nabla_b.begin(), nabla_b.end(), [](MatrixXd& mat){mat.setZero();});

    // Feed forward to get the activations in each layer
    VectorXd activation = training_example.first;

    VectorList activations;
    VectorList z_vectors;

    activations.push_back(activation);

    for (int i = 0; i < num_layers_ - 1; ++i) {
        auto z = weights_[i] * activation + biases_[i];

        z_vectors.push_back(z);
        activation = sigmoid_vec(z);
        activations.push_back(activation);
    }

    // Get initial error in final layer
    VectorXd delta = cost_derivative(activations.back(), 
        training_example.second).cwiseProduct(sigmoid_prime_vec(z_vectors.back()));
    nabla_b.back() = delta;
    nabla_w.back() = delta * activations[activations.size() - 2].transpose();

    // Backwards pass to get the error and gradient in each preceding layer
    for (int i = num_layers_ - 2; i >= 1; --i) {
        auto z = z_vectors[i-1];
        auto spv = sigmoid_prime_vec(z);

        delta = (weights_[i].transpose() * delta).cwiseProduct(spv);

        nabla_b[i-1] = delta;
        nabla_w[i-1] = delta * activations[i-1].transpose();
    }

    return std::make_pair(nabla_b, nabla_w);
}

VectorXd BasicNetwork::cost_derivative(const VectorXd& output_activations,
    const VectorXd& y) {

    return output_activations - y;
}

VectorXd BasicNetwork::feed_forward(VectorXd a) {
    for (int i = 0; i < weights_.size(); i++) {
        a = sigmoid_vec(weights_[i] * a + biases_[i]);
    }
    return a;
}

size_t BasicNetwork::evaluate(std::vector<DataPair>& test_data) {

    size_t examples_correct = 0;

    for (auto& pair : test_data) {
        auto input = pair.first;
        auto label = pair.second;

        bool result = false;
        auto output = feed_forward(input);

        double max_coefficient = output.maxCoeff();
        for (int i = 0; i < output.size(); ++i) {
            if (output(i) == max_coefficient) {
                result = (label(i) == 1.0);
                break;
            }
        }
        if (result) {
            examples_correct++;
        }
    }
    return examples_correct;
}

double BasicNetwork::sigmoid_func(double z) {
    return 1/(1 + std::exp(-z));
}

VectorXd BasicNetwork::sigmoid_vec(const VectorXd& z) {
    return z.unaryExpr([](double arg) { 
        return BasicNetwork::sigmoid_func(arg);
    });
}

VectorXd BasicNetwork::sigmoid_prime_vec(const VectorXd& z) {
    return z.unaryExpr([](double arg) { 
        double sigz = BasicNetwork::sigmoid_func(arg);
        return -sigz/(1 - sigz);
    });
}
