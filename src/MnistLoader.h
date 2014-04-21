#include <string>
#include <vector>

#include "Util.h"

using namespace Eigen;

class MnistLoader {
public:
    std::pair<Dataset, Dataset> load();
private:
    int reverse_int(int i);

    Dataset load_dataset(
        const std::string& image_path,
        const std::string& label_path);

    VectorList load_images(const std::string& path);
    VectorList load_labels(const std::string& path);
};
