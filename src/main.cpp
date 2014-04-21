#include <iostream>

#include "BasicNetwork.h"
#include "MnistLoader.h"

int main(int argc, const char *argv[])
{
    MnistLoader data_loader;
    auto mnist_dataset = data_loader.load();

    BasicNetwork network({784, 20, 10});

    return 0;
}
