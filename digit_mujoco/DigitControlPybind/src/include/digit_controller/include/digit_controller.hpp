#pragma once

#include "analytical_expressions.hpp"
#include "digit_definition.hpp"
#include "digit_state.hpp"
#include "domain_control.hpp"
#include "geo_expressions.hpp"
#include "motor_control.hpp"
#include "robot_expressions.hpp"
#include "safety_manager.hpp"
#include "walking_trajectory.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include "BasicEigenTypes.hpp"

enum ControlMode { // Remove this and use domain control.
    ZERO_POSITION,
    STANDING,
    WALKING_GAIT,
    HANGING
};

namespace digitcontroller{
    class DigitController {
    public:
        // constructor and destructor and reseet
        explicit DigitController(const std::string& path2toml_in, double control_dt_in);

        ~DigitController();

        void reset();

        void walkingCommandCompute();

//        void hangingCommandCompute();
        void computeTorque(Ref<EigenVec> torqueOut, Ref<EigenVec> velReferenceOut);
        void setUsrCommand(const Ref<EigenVec> &usrCommand);
        void setStates(const Ref<EigenVec> &digitStates);
        double getPhaseVariable(){return phase_variable;}
        DomainControl::DomainType getDomain(){return domain;}
        bool getDomainSwitch(){return domain_switch;}

    private:
        double control_dt;
        double desired_height;
        // size of control inputs
        const int N_joint;
        const int N_motor;

        // Torque commands are user-generated in this context
        VectorXd command;  //commanded torque for actuated joints
        VectorXd ff_command;
        VectorXd fb_command;

        VectorXd pos_reference;
        VectorXd vel_reference;
        VectorXd acc_reference;

        VectorXd position_acj;
        VectorXd velocity_acj;


        DigitState currState;

        VectorXd dGains; // ZG: Check damping and dGains difference.
        VectorXd damping;

        bool isReset = false;
        bool domain_switch = false;
        double phase_variable = 0;
        double time_in_current_mode = 0;

        DomainControl::DomainType domain = DomainControl::DomainType::DOUBLE_SUPPORT;
        double target_vel_x_raw = 0; //-0.09;
        double target_vel_y_raw = 0;
        double target_yaw = 0.0;
        double ang_vel = 0.0;

        AnalyticalExpressions analytical_expressions;

        // Initialize the hyper_parameter with default step time and
        // step width (used by foot placement controller).
        // Target velocity is given by PSP or user.
        DigitControlHyperparameters hyper_param;

        std::string path2toml;

        SafetyManager SafetyLayer;
        MotorControl PDLayer;
        WalkingTrajectory TrajLayer;

        VectorXd q_B;
        // body position
//        ControlMode cm = ControlMode::WALKING_GAIT;

        // internal control state variables
        VectorXd usrCommand_;
        VectorXd digit_state_;
        VectorXd base_position_;
        VectorXd base_quaternion_;
        VectorXd base_velocity_;
        VectorXd base_angvel_;
        VectorXd position_full_;
        VectorXd velocity_full_;
    };
}
