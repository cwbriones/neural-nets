#include <iostream>
#include <fstream>

#include "MnistLoader.h"

std::pair<Dataset, Dataset> MnistLoader::load(const int training_size, const int test_size) {
    std::string training_path   = "./data/train-images-idx3-ubyte";
    std::string training_labels = "./data/train-labels-idx1-ubyte";

    std::string test_path   = "./data/t10k-images-idx3-ubyte";
    std::string test_labels = "./data/t10k-labels-idx1-ubyte";

    std::cout << "Loading MNIST datasets." << std::endl;
    std::cout << "=======================" << std::endl;

    std::cout << "Loading training data." << std::endl;
    auto training_data = load_dataset(training_path, training_labels, training_size);
    std::cout << training_data.size() << " examples loaded." << std::endl;
    std::cout << std::endl;

    std::cout << "Loading test data." << std::endl;
    auto test_data = load_dataset(test_path, test_labels, test_size);
    std::cout << test_data.size() << " examples loaded." << std::endl;
    std::cout << std::endl;

    std::cout << "Completed." << std::endl;
    return std::make_pair(training_data, test_data);
}

int MnistLoader::reverse_int(int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return static_cast<int>(c1 << 24) + static_cast<int>(c2 << 16) + static_cast<int>(c3 << 8) + c4;
}

VectorList MnistLoader::load_images(const std::string& path, const size_t count) {
    std::vector<VectorXd> images;

    std::ifstream file(path, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int rows = 0;
        int cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number)); 
        magic_number = reverse_int(magic_number);

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        if (count > 0 && count < number_of_images) {
            number_of_images = count;
        }

        file.read((char*)&rows, sizeof(rows));
        rows = reverse_int(rows);

        file.read((char*)&cols,sizeof(cols));
        cols = reverse_int(cols);

        for (int i = 0; i < number_of_images; ++i) {
            // Read in each image as a 784 (28 x 28) element vector
            VectorXd image_data(rows * cols);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    unsigned char tmp = 0;
                    file.read((char*)&tmp, sizeof(tmp));

                    image_data(r * rows + c) = static_cast<double>(tmp)/255.0;
                }
            }
            images.push_back(image_data);
        }
    } else {
        std::cerr << "ERROR: File " << path << " not found." << std::endl;
    }

    return images;
}

VectorList MnistLoader::load_labels(const std::string& path, const size_t count) {
    std::ifstream file(path, std::ios::binary);
    std::vector<VectorXd> labels;

    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_labels = 0;

        file.read((char*)&magic_number, sizeof(magic_number)); 
        magic_number = reverse_int(magic_number);

        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverse_int(number_of_labels);
        if (count > 0 && count < number_of_labels) {
            number_of_labels = count;
        }

        for (int i = 0; i < number_of_labels; ++i) {
            // Read in each label as a 10 element unit vector
            // where the index k = 1 if k == label otherwise 0
            unsigned char k = 0;
            file.read((char*)&k, sizeof(k));

            VectorXd label(10);
            label.setZero();
            label(k) = 1.0;

            labels.push_back(label);
        }
    } else {
        std::cerr << "ERROR: File " << path << " not found." << std::endl;
    }

    return labels;
}

Dataset MnistLoader::load_dataset(
    const std::string& image_path,
    const std::string& label_path,
    const int size) {

    std::cout << "Loading images..." << std::endl;
    auto images = load_images(image_path, size);

    std::cout << "Loading labels..." << std::endl;
    auto labels = load_labels(label_path, size);

    Dataset dataset;
    for (int i = 0; i < images.size(); ++i) {
        dataset.push_back(std::make_pair(images[i], labels[i]));
    }

    return dataset;
}
