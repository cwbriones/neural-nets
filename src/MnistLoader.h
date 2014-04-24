#include <string>
#include <vector>

#include "Util.h"

using namespace Eigen;

class MnistLoader {
public:
    std::pair<Dataset, Dataset> load(const int training_size = -1, const int test_size = -1);
private:
    int reverse_int(int i);

    Dataset load_dataset(
        const std::string& image_path,
        const std::string& label_path,
        const int size = -1);

    VectorList load_images(const std::string& path, const size_t count = -1);
    VectorList load_labels(const std::string& path, const size_t count = -1);
};
