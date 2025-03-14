//
// Created by hyunyoungjung on 8/8/23.
//

#ifndef DIGIT_WS_DIGITVEC_H
#define DIGIT_WS_DIGITVEC_H
#include "omp.h"
#include "Yaml.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <filesystem>
#include "BasicEigenTypes.hpp"
#include "digit_controller.hpp"

class digitControlEnv {
    public:
    explicit digitControlEnv(double control_dt_) {
        control_dt = control_dt_;
    }

    ~digitControlEnv() {
        delete controller_;
    }
    void init(){
        // Use the __FILE__ macro to get the current file's path
        std::filesystem::path currentFilePath = __FILE__;

        // Convert to an absolute path
        std::filesystem::path absolutePath = std::filesystem::absolute(currentFilePath);

        // Convert the absolute path to a std::string
        std::string absolutePathStr = absolutePath.string();

        size_t pos = absolutePathStr.find_last_of('/');
        if (pos != std::string::npos) {
            absolutePathStr = absolutePathStr.substr(0, pos + 1);  // include the trailing '/'
        }

        std::string suffix = "include/digit_controller/src";

        controller_ = new digitcontroller::DigitController(absolutePathStr + suffix, control_dt);
    }

    void reset(){
        controller_->reset();
    }

    void setStates(Ref<EigenRowMajorMat> &digitStates){
        controller_->setStates(digitStates.row(0));
    }

    void computeTorque(Ref<EigenRowMajorMat> &torqueOut, Ref<EigenRowMajorMat> &velReferenceOut){
        controller_->computeTorque(torqueOut.row(0), velReferenceOut.row(0));
    }
    void setUsrCommand(Ref<EigenRowMajorMat> &usrCommand){
        controller_->setUsrCommand(usrCommand.row(0));
    }
    double getPhaseVariable(){return controller_->getPhaseVariable();}
    int getDomain(){return controller_->getDomain();}
    bool getDomainSwitch(){return controller_->getDomainSwitch();}
private:
    digitcontroller::DigitController * controller_;
    double control_dt;
};
#endif //DIGIT_WS_DIGITVEC_H
