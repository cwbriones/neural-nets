#include <iostream>

#include "BasicNetwork.h"
#include "MnistLoader.h"

int main(int argc, const char *argv[])
{
    MnistLoader data_loader;
    auto mnist_dataset = data_loader.load();

    auto training_data = mnist_dataset.first;
    auto test_data = mnist_dataset.second;

    BasicNetwork network({784, 30, 10});

    network.TrainSGD(training_data, 30, 10, 0.3, test_data);

    return 0;
}
