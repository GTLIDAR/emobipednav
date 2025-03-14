//
// Created by jemin on 1/13/21.
//

#ifndef _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_BASICEIGENTYPES_HPP_
#define _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_BASICEIGENTYPES_HPP_


using Dtype = float;
using EigenRowMajorMat = Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec = Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec = Eigen::Matrix<bool, -1, 1>;
using EigenIntVec = Eigen::Matrix<int, -1, 1>;
using EigenDoubleVec = Eigen::Matrix<double, -1, 1>;


#endif //_RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_BASICEIGENTYPES_HPP_
