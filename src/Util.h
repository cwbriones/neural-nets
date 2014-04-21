#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>

// Type representing an entry of training data.
//
// The first vector is the input to the network and the 
// second vector is the desired output.
typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> DataPair;
typedef std::vector<DataPair> Dataset;

typedef std::vector<Eigen::VectorXd> VectorList;
typedef std::vector<Eigen::MatrixXd> MatrixList;

#endif
