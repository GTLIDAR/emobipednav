//
// Created by hyunyoungjung on 7/31/23.
//

#include "digit_controller.hpp"


namespace digitcontroller {

    DigitController::DigitController(const std::string& path2toml_in, double control_dt_in )
    : N_joint(wdc::joint::N_JOINTS), N_motor(wdc::motor::N_MOTORS),
    hyper_param(0.4, 0.4, Vector3d::Zero(), 0, 0.4),
      SafetyLayer(path2toml_in), PDLayer(jointControl::PD_CONTROL, path2toml_in),
      TrajLayer(path2toml_in)
    {
            control_dt = control_dt_in;
            // Torque commands are user-generated in this context
            command = VectorXd::Zero(N_motor); // commanded torque for actuated joints
            ff_command = VectorXd::Zero(N_motor);
            fb_command = VectorXd::Zero(N_motor);

            pos_reference = VectorXd::Zero(N_motor);
            vel_reference = VectorXd::Zero(N_motor);
            acc_reference = VectorXd::Zero(N_motor);

            position_acj = VectorXd ::Zero(N_motor);
            velocity_acj = VectorXd ::Zero(N_motor);

            dGains = VectorXd::Zero(N_motor); // ZG: Check damping and dGains difference.
            damping = VectorXd::Zero(N_motor);

            path2toml = path2toml_in;

            TrajLayer.changeDomain(DomainControl::DomainType::RIGHT_STAND); //HW: refer to domain_control.hpp for details
            command.setZero(N_motor);
            damping << 66.849, 26.1129, 38.05, 38.05, 15.5532, 15.5532, 
                        66.849, 26.1129, 38.05, 38.05, 15.5532, 15.5532, 
                        66.849, 66.849, 26.1129, 66.849, 
                        66.849, 66.849, 26.1129, 66.849;
            // Set initial mode to be standing. Since I don't change the mode, this should stay constant
            SafetyLayer.buttons.buttons[button::REMOTE_STANDSTILL].isPressed = false;
            SafetyLayer.buttons.buttons[button::REMOTE_WALK].isPressed = true;
            SafetyLayer.buttons.buttons[button::REMOTE_ZERO_POSITION].isPressed = false;            
        }

    DigitController::~DigitController() = default;

    void DigitController::reset(){
        // get first state and initialize
        time_in_current_mode = 0;
        phase_variable = 0;
        domain = DomainControl::DomainType::DOUBLE_SUPPORT;
        target_vel_x_raw = 0; //-0.09;
        target_vel_y_raw = 0;
        target_yaw = 0.0;
        ang_vel = 0.0;
        isReset = true;
    }


    void DigitController::walkingCommandCompute() {

            // Create state q_B that is in Frost definition order so that it can be used in Frost.
            // I thinks Frost is used to compute kinematics of robot in body frame
            // https://github.gatech.edu/GeorgiaTechLIDARGroup/Digit_CFROST
            // q_b has three global positions, x, y, and z
            // three global orientations, yaw, pitch, and roll
            // 20 positions of all active joints
            // 2 positions of one type of passive joint, tarsus joint
            VectorXd q_B = DigitState::buildState(currState.base_position,
                                                  Vector3d{currState.base_orientation(0), currState.base_orientation(1), currState.base_orientation_unfil(2)},
                                                  currState.position_full);
            q_B.head(4) = VectorXd::Zero(4); // reset x, y, z, and yaw as zero
            target_yaw += ang_vel * control_dt; // update target yaw every time step
            if (target_yaw > M_PI){
                target_yaw -= 2.0 * M_PI;
            }
            else if (target_yaw < -M_PI){
                target_yaw += 2.0 * M_PI;
            }

            //==============Contact Guard==========//
            if (domain_switch) {
                // Return true at every contact with ground; or
                // switch from DOUBLE_SUPPORT to RIGHT_STAND.
                // note that domain is double support at the beggining

                TrajLayer.updateTimeOffset(time_in_current_mode);

                // When switching, smooth torque using polynomial interpolation.
                MatrixXd p_BToe_l = analytical_expressions.p_left_toe_pitch(q_B);
                MatrixXd p_BToe_r = analytical_expressions.p_right_toe_pitch(q_B);
                MatrixXd p_BC = analytical_expressions.p_COM(q_B);

                TrajLayer.recordVelocityTarget();
                TrajLayer.is_lateral_v_reachable = true;
                TrajLayer.fail_mode = 0;

                // Record state before contact in current frame.
                TrajLayer.v_StC_0 =
                        TrajLayer.v_StB_Wpc_L; // Current body velocity right after contact.
                TrajLayer.v_StC_0_d =
                        TrajLayer.v_StB_f; // Body velocity right before contact.
                TrajLayer.p_StC_0_d = TrajLayer.p_SwC_St_df;                

                // Set time only once before each step. Then inside the step
                // the time is updated by the controller.
                TrajLayer.setStepTime(hyper_param.step_time_crnt,
                                        hyper_param.step_time_next);

                // Record pre-contact state in next frame.
                double yaw_StSw_d_F0 = TrajLayer.yaw_StSw_df;
                Matrix3d R_StSw_d_F0 =
                        AngleAxisd(yaw_StSw_d_F0, Vector3d::UnitZ()).toRotationMatrix();
                TrajLayer.v_StC_0 = R_StSw_d_F0.transpose() * TrajLayer.v_StC_0;
                TrajLayer.v_StC_0_d = R_StSw_d_F0.transpose() * TrajLayer.v_StC_0_d;
                TrajLayer.p_StC_0_d = R_StSw_d_F0.transpose() * TrajLayer.p_StC_0_d;

                if (domain == DomainControl::DomainType::RIGHT_STAND) { // right foot in stance
                    TrajLayer.p_StC_0 = -p_BToe_r + p_BC;
                    TrajLayer.p_BSw_initial = p_BToe_l;

                    TrajLayer.updateYawTarget(
                            target_yaw, currState.base_orientation_unfil(2),
                            currState.position_full(wdc::joint::RIGHT_HIP_YAW),
                            currState.position_full(wdc::joint::LEFT_HIP_YAW));
                } else { // left foot in stance
                    TrajLayer.p_StC_0 = -p_BToe_l + p_BC;
                    TrajLayer.p_BSw_initial = p_BToe_r;

                    TrajLayer.updateYawTarget(
                            target_yaw, currState.base_orientation_unfil(2),
                            currState.position_full(wdc::joint::LEFT_HIP_YAW),
                            currState.position_full(wdc::joint::RIGHT_HIP_YAW));
                }
                TrajLayer.roll_df *= 0.5; // Decay the body roll to zero.

                // time_elapsed = (std::chrono::system_clock::now() -
                // time_walking).count(); TrajLayer.logStepTime(time_elapsed);
                TrajLayer.toe_off = false;
                TrajLayer.s_toe_off = 0;
                TrajLayer.touch_down = false;
                TrajLayer.s_touch_down = 0;
            }

            phase_variable = TrajLayer.compute_phaseVariable(time_in_current_mode);

            // regulation_PSP includes swing and stance foot control (modifies command
            // and pos_ref)
            TrajLayer.regulation_PSP(currState, pos_reference, vel_reference,
                                        phase_variable, hyper_param, desired_height, acc_reference);

            fb_command =
                    PDLayer.robotControl(pos_reference, position_acj, vel_reference,
                                            velocity_acj, dGains, domain);

            // Calculate the accelerate induced torque.
            ff_command = TrajLayer.calc_passivity_command(
                    currState, pos_reference, vel_reference, velocity_acj, acc_reference);

            command = fb_command + ff_command;
        }
    
    void DigitController::setStates(const Ref<EigenVec> &digitStates){
        // root_pos,  # 2 0~3
        // self.root_quat,  # 4 3~7
        // self.root_lin_vel,  # 3 7~10, in the world frame
        // self.root_ang_vel,  # 3 10~13
        // self.motor_pos,  # 20 13~33
        // self.passive_hinge_pos,  # 10 33~43
        // self.motor_vel,  # 20     43~63
        // self.passive_hinge_vel  # 10 63~73
        digit_state_ = digitStates.cast<double>();
        base_position_ = digit_state_.segment(0, 3);
        base_quaternion_ = digit_state_.segment(3, 4);
        base_velocity_ = digit_state_.segment(7, 3);
        base_angvel_ = digit_state_.segment(10, 3);
        position_full_ = digit_state_.segment(13, N_joint);
        velocity_full_ = digit_state_.segment(13 + N_joint, N_joint);

        if (isReset) {
            isReset = false;
            currState = SafetyLayer.getFullFeedback(velocity_full_, position_full_,
                                                    base_position_, base_velocity_,
                                                    base_angvel_, base_quaternion_);
            
            VectorXd q_0 = DigitState::buildState(VectorXd::Zero(3), VectorXd::Zero(3), currState.position_full);

            MatrixXd p_PTL_0 = analytical_expressions.p_left_toe_pitch(q_0);
            MatrixXd p_PTR_0 = analytical_expressions.p_right_toe_pitch(q_0);
            MatrixXd p_MP_P_0 =  -((p_PTL_0 + p_PTR_0)/2);
            // desired_height = p_MP_P_0(2);
            desired_height = 0.969116; // TODO: check if it is correct

            TrajLayer.reloadRegulators(path2toml, true); // load regulator values for online implementation
            target_yaw = currState.base_orientation_unfil(2);
            TrajLayer.tg_yaw = target_yaw; // Hack! To initialize the member variable
            // tg_yaw and tg_yaw_old.
            TrajLayer.updateYawTarget(
                target_yaw, currState.base_orientation_unfil(2),
                currState.position_full(wdc::joint::RIGHT_HIP_YAW),
                currState.position_full(wdc::joint::LEFT_HIP_YAW));
        }

        domain_switch = TrajLayer.evalGuardSensor(currState, domain, phase_variable);
    }
    
    void DigitController::setUsrCommand(const Ref<EigenVec> &usrCommand){
        usrCommand_ = usrCommand.cast<double>();
        target_vel_x_raw = usrCommand_[0];
        target_vel_y_raw = usrCommand_[1];
        hyper_param.v_StB_d.head(2) = Vector2d{target_vel_x_raw, target_vel_y_raw};
        ang_vel = usrCommand_[2];
    }

    void DigitController::computeTorque(Ref<EigenVec> torqueOut, Ref<EigenVec> velRefenceOut) {
        // switch mode depending on the command
        // currState is a post-process states
        // if using no filter, it's same as digitStates
        currState = SafetyLayer.getFullFeedback(velocity_full_, position_full_,
                                                base_position_, base_velocity_,
                                                base_angvel_, base_quaternion_);

        position_acj = currState.position_full.head(N_motor);
        velocity_acj = currState.velocity_full.head(N_motor);

        // comptue command
        walkingCommandCompute();
        time_in_current_mode += control_dt;
        torqueOut = SafetyLayer.clampCommand(command).cast<float>();
        velRefenceOut = vel_reference.cast<float>();
    }
} // namespace digitcontroller
