/*
* Copyright (c) 2023, The Ohio State University - Cyberbotics Lab
* All rights reserved.
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/**
* Author:       Victor Paredes, Guillermo Castillo
* Email:        paredescauna.1@osu.edu, castillomartinez.2@osu.edu
* Modified:     01-25-2022
* Copyright:    Cyberbotics Lab @The Ohio State University
**/
/**
* Author:       Zhaoyuan Gu, Aziz Shamsah
* Email:        {zgu78, ashamsah3}@gatech.edu
* Modified:     02-25-2023
* Copyright:    LIDAR @ Georgia Tech
**/
#include "safety_manager.hpp"
#include <iostream>
#include <unistd.h>
#include "utilities.hpp"

using namespace wdc;

struct JointLimits{
  // It is safer to assume 0 by default
  float position_min = 0;    
  float position_max = 0;
  float velocity_max = 0;
};

SafetyLimits::JointLimits LHipRoll = {-60, 60, 60};
SafetyLimits::JointLimits LHipYaw = {-40, 40, 60};
SafetyLimits::JointLimits LHipPitch = {-60, 90, 60};
SafetyLimits::JointLimits LKnee = {-80, 58.4, 60};
SafetyLimits::JointLimits LShin = { -100, 100, 60}; //Not limited
SafetyLimits::JointLimits LTarsus = {-50.3, 71.6, 60};
SafetyLimits::JointLimits LHeelSpring = { -20, 20, 60};   //Real limit is 6, but the real value violates that limit, hence used 20
SafetyLimits::JointLimits LToeA = {-46.2755, 44.9815, 60};
SafetyLimits::JointLimits LToeB = {-45.8918, 45.5476, 60};
SafetyLimits::JointLimits LToePitch = {-44, 34, 60};
SafetyLimits::JointLimits LToeRoll = {-37, 33, 60};
SafetyLimits::JointLimits LShoulderRoll = {-75, 75, 60};
SafetyLimits::JointLimits LShoulderPitch = {-145, 145, 60};
SafetyLimits::JointLimits LShoulderYaw = {-100, 100, 60};
SafetyLimits::JointLimits LElbow = {-77.5, 77.5, 60};

SafetyLimits::JointLimits RHipRoll = {-60, 60, 60};
SafetyLimits::JointLimits RHipYaw = {-40, 40, 60};
SafetyLimits::JointLimits RHipPitch = {-90, 60, 60};
SafetyLimits::JointLimits RKnee = {-58.4, 80, 60};
SafetyLimits::JointLimits RShin = { -100, 100, 60}; //Not limited
SafetyLimits::JointLimits RTarsus = {-71.6, 50.3, 60};
SafetyLimits::JointLimits RHeelSpring = { -20, 20, 60};
SafetyLimits::JointLimits RToeA = {-44.9815, 46.2755, 60};
SafetyLimits::JointLimits RToeB = {-45.5476, 45.8918, 60};
SafetyLimits::JointLimits RToePitch = {-34, 44, 60};
SafetyLimits::JointLimits RToeRoll = {-33, 37, 60};
SafetyLimits::JointLimits RShoulderRoll = {-75, 75, 60};
SafetyLimits::JointLimits RShoulderPitch = {-145, 145, 60};
SafetyLimits::JointLimits RShoulderYaw = {-100, 100, 60};
SafetyLimits::JointLimits RElbow = {-77.5, 77.5, 60};

// Angle direction is the same for a joint independently if left or right
// Structure of the joints order is [Left actuated joints, Right actuated joints, Left unactuated joints, Right unactuated joints]
SafetyLimits::JointLimits JointList[] = {
 LHipRoll, LHipYaw, LHipPitch, LKnee, LToeA, LToeB,
 RHipRoll, RHipYaw, RHipPitch, RKnee, RToeA, RToeB,
 LShoulderRoll, LShoulderPitch, LShoulderYaw, LElbow,
 RShoulderRoll, RShoulderPitch, RShoulderYaw, RElbow,
 LShin, LTarsus, LToePitch, LToeRoll, LHeelSpring,
 RShin, RTarsus, RToePitch, RToeRoll, RHeelSpring
};

// Command is given in N/m
SafetyLimits::ActuatorLimits HipRoll = {126};
SafetyLimits::ActuatorLimits HipYaw = {79};
SafetyLimits::ActuatorLimits HipPitch = {216};
SafetyLimits::ActuatorLimits Knee = {231}; 
SafetyLimits::ActuatorLimits ToeA = {41};
SafetyLimits::ActuatorLimits ToeB = {41};
SafetyLimits::ActuatorLimits ShoulderRoll = {126};
SafetyLimits::ActuatorLimits ShoulderPitch = {126};  
SafetyLimits::ActuatorLimits ShoulderYaw = {79};
SafetyLimits::ActuatorLimits Elbow = {126};

SafetyLimits::ActuatorLimits ActuatorList[] = {
  HipRoll, HipYaw, HipPitch, Knee, ToeA, ToeB, 
  HipRoll, HipYaw, HipPitch, Knee, ToeA, ToeB, 
  ShoulderRoll, ShoulderPitch, ShoulderYaw, Elbow, 
  ShoulderRoll, ShoulderPitch, ShoulderYaw, Elbow 
};


SafetyManager::SafetyManager(std::string path2toml) {

  // FILTERING
  config = cpptoml::parse_file(path2toml + "/config_files/robot_config.toml");
  // config = cpptoml::parse_file("robot_config.toml");
  if(config->get_qualified_as<std::string>("filter.use_filter").value_or("false") == "yes") {
    isUsingFilter = true;
    size_windowFilter = config->get_qualified_as<int>("filter.window_size").value_or(1);
  }else{
    isUsingFilter = false;
  }

  if(isUsingFilter) {
    for(int i = 0; i < N_JOINTS; i++) {
      filteredJointVelocity[i] = JointFilter(size_windowFilter);
    }
    for(int i = 0; i < 3; i++) {
      filteredBaseAngVel[i] = JointFilter(size_windowFilter);
      filteredBaseVel[i] = JointFilter(size_windowFilter);
    }    
  }  

  filteredRoll = JointFilter(size_windowFilter);
  filteredPitch = JointFilter(size_windowFilter);
  filteredYaw = JointFilter(size_windowFilter);

}
int SafetyManager::updateYawOffset(double newYawOffset)
{
  yawOffset = newYawOffset;
  return 0;
}

VectorXd SafetyManager::clampCommand(VectorXd command) {
  for (int i = 0; i < motor::N_MOTORS; i++) {
    if (command[i] > ActuatorList[i].torque_max) {
      command[i] = ActuatorList[i].torque_max;
    } else if (command[i] < -ActuatorList[i].torque_max) {
      command[i] = -ActuatorList[i].torque_max;
    }
  }
  return command;
}


struct EulerAngles {
    double roll, pitch, yaw;
};

// ZG: This is same as wikipedia's conversion formula, which is quaternion to ZYX intrinsic.
EulerAngles ToEulerAngles(double qw, double qx, double qy, double qz) {
    EulerAngles angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (qw * qx + qy * qz);
    double cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (qw * qz + qx * qy);
    double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    // [-pi, pi]. https://en.cppreference.com/w/cpp/numeric/math/atan2
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}

DigitState SafetyManager::getFullFeedback(const Ref<VectorXd>& velocity_full_in,
                                          const Ref<VectorXd>& position_full_in,
                                          const Ref<VectorXd>& base_position_in,
                                          const Ref<VectorXd>& base_velocity_in,
                                          const Ref<VectorXd>& base_angvel_in,
                                          const Ref<VectorXd>& base_quaternion_in
                                          ) {
    DigitState newState;

    // joint state update
    // first 20: motor vel, second 2
    for(int32_t i = 0; i < wdc::joint::N_JOINTS; i++) {
        if (i < motor::N_MOTORS)  // the first motor::N_MOTORS are active joints (with motors)
        {
            if(isUsingFilter) { // seems that for this version, we did not use filter.
                filteredJointVelocity[i].addPoint(velocity_full_in[i]);
                newState.velocity_full[i] = filteredJointVelocity[i].getFilteredData();
            }else{
                newState.velocity_full[i] = velocity_full_in[i];
            }

            newState.position_full[i] = position_full_in[i];

        }else // for passive joints -> no torque
        {
            if(isUsingFilter) { // seems that for this version, we did not use filter.
                filteredJointVelocity[i].addPoint(velocity_full_in[i]);
                newState.velocity_full[i] = filteredJointVelocity[i].getFilteredData();
            }else{
                newState.velocity_full[i] = velocity_full_in[i];
            }
            newState.position_full[i] = position_full_in[i];
        }
    }

    // base state update
    for(int32_t i = 0; i < 3; i++) {
        newState.base_position[i] = base_position_in[i];
        newState.IMU_angvel[i] = 0; // no need IMU in simulation

        if(isUsingFilter) {
            filteredBaseVel[i].addPoint(base_velocity_in[i]);
            filteredBaseAngVel[i].addPoint(base_angvel_in[i]);
            newState.base_velocity[i] = filteredBaseVel[i].getFilteredData();
            newState.base_angvel[i] = filteredBaseAngVel[i].getFilteredData();
            newState.IMU_acceleration[i] = 0;
        }else{
            newState.base_velocity[i] = base_velocity_in[i];
            newState.base_angvel[i] = base_angvel_in[i];
            newState.IMU_acceleration[i] = 0;
        }
    }

    newState.IMU_quaternion << 0.,0.,0.,0; // no need IMU in simulation
    newState.base_quaternion = base_quaternion_in;  //Load measurement from API. Format: [w,x,y,z]

    EulerAngles angles;
  angles = ToEulerAngles(newState.base_quaternion[0],      //base quaterion Format: [w, x, y, z]
                         newState.base_quaternion[1],
                         newState.base_quaternion[2],
                         newState.base_quaternion[3]);

  newState.base_orientation_unfil << angles.roll, angles.pitch, angles.yaw;

  if(isUsingFilter) {
    filteredRoll.addPoint(angles.roll);
    angles.roll = filteredRoll.getFilteredData();
    filteredPitch.addPoint(angles.pitch);
    angles.pitch = filteredPitch.getFilteredData();
    filteredYaw.addPoint(angles.yaw);
    angles.yaw = filteredYaw.getFilteredData() - yawOffset;
  }
  newState.base_orientation << angles.roll, angles.pitch, angles.yaw;
  return newState;
}
