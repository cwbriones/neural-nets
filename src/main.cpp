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

    BasicNetwork network({784, 20, 20, 10});

    const int EPOCHS = 30;
    const int MINI_BATCH_SIZE = 10;
    const double ETA = 0.3;

    network.TrainSGD(training_data, EPOCHS, MINI_BATCH_SIZE, ETA, test_data);
    return 0;
}
