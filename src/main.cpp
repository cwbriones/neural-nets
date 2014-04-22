#include <iostream>

#include "BasicNetwork.h"
#include "MnistLoader.h"

using namespace Eigen;

int main(int argc, const char *argv[])
{
    MnistLoader data_loader;
    auto mnist_dataset = data_loader.load();

    auto training_data = mnist_dataset.first;
    auto test_data = mnist_dataset.second;

    BasicNetwork network({784, 30, 10});

    const int EPOCHS = 1;
    const int MINI_BATCH_SIZE = 30;
    const double ETA = 0.3;

    network.TrainSGD(training_data, EPOCHS, MINI_BATCH_SIZE, ETA, test_data);
    size_t num_correct = network.evaluate(test_data);

    std::cout << "Accuracy after " << EPOCHS << " epochs: " 
              << num_correct << "/" << test_data.size() << std::endl;
    return 0;
}
