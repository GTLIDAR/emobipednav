cmake_minimum_required(VERSION 3.14)
project(_Digit_IFM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DigitControlPybind/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DigitControlPybind/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DigitControlPybind/bin)

set(LOCAL_INCL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DigitControlPybind/src/include/digit_controller/include)
set(LOCAL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DigitControlPybind/src/include/digit_controller/src)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DigitControlPybind/src/thirdParty/pybind11 pybind11)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/DigitControlPybind/src/thirdParty/Eigen3)
find_package(Eigen3 CONFIG REQUIRED)
#find_package(OpenMP REQUIRED)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

#######################################
# CMake for digit controller
#######################################
include_directories(
        ${LOCAL_INCL_DIR}/
        ${LOCAL_INCL_DIR}/MathExpr/
        ${LOCAL_INCL_DIR}/Eigen/
        ${LOCAL_INCL_DIR}/GeometricKinematics
        ${LOCAL_INCL_DIR}/AnalyticalKinematicsDynamics
)
# GeometricKinematics
add_library(geometric_kinematics_lib
        ${LOCAL_INCL_DIR}/GeometricKinematics/IK_Geo_p.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/IK_Geo_v.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/J_IK_Geo_p.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/dJ_IK_Geo_p.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/FK_Geo_p.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/FK_Geo_v.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/J_FK_Geo_p.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/dJ_FK_Geo_p.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/KinGeo_rtwutil.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/KinGeo_initialize.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/rt_nonfinite.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/rtGetNaN.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/rtGetInf.cpp
        ${LOCAL_INCL_DIR}/GeometricKinematics/geo_expressions.cpp
)
target_compile_options(geometric_kinematics_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

# Analytical Kinematics and Dynamics
file(GLOB_RECURSE KIN_DYN_SRCS 
  ${LOCAL_INCL_DIR}/AnalyticalKinematicsDynamics/kin/*.cpp # contact points are fixed-arm version in this folder.
  ${LOCAL_INCL_DIR}/AnalyticalKinematicsDynamics/dynFixArm/*.cpp # Use fixed-arm dynamics
)
add_library(kinematics_dynamics_lib
  ${KIN_DYN_SRCS}
  ${LOCAL_INCL_DIR}/AnalyticalKinematicsDynamics/analytical_expressions.cpp
)
target_include_directories(kinematics_dynamics_lib PUBLIC
        ${LOCAL_INCL_DIR}/AnalyticalKinematicsDynamics/kin
        ${LOCAL_INCL_DIR}/AnalyticalKinematicsDynamics/dynFixArm
)
target_compile_options(kinematics_dynamics_lib PRIVATE -fPIC -O3 -flto -mno-avx2)


# Digit Controller
set(EIGEN_INCL ${LOCAL_INCL_DIR}/Eigen)
set(TOML_INCL ${LOCAL_INCL_DIR}/cpptoml)


set(EXPR_SRCS
  ${LOCAL_INCL_DIR}/MathExpr/J_leftFoot.cc
  ${LOCAL_INCL_DIR}/MathExpr/J_rightFoot.cc
  ${LOCAL_INCL_DIR}/MathExpr/p_leftFoot.cc
  ${LOCAL_INCL_DIR}/MathExpr/p_rightFoot.cc
  ${LOCAL_INCL_DIR}/MathExpr/p_com.cc
  ${LOCAL_INCL_DIR}/MathExpr/v_com.cc
  ${LOCAL_INCL_DIR}/MathExpr/T_leftFoot.cc
  ${LOCAL_INCL_DIR}/MathExpr/T_rightFoot.cc
)

set(EXPR_INCL ${LOCAL_INCL_DIR}/MathExpr)

# Create core libraries for cyberbotics lab
add_library(joint_filter_lib STATIC ${LOCAL_SRC_DIR}/filter.cpp)
target_include_directories(joint_filter_lib PUBLIC ${LOCAL_INCL_DIR} ${EIGEN_INCL})
target_link_libraries(joint_filter_lib)
target_compile_options(joint_filter_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

add_library(safety_manager_lib STATIC ${LOCAL_SRC_DIR}/safety_manager.cpp)
target_include_directories(safety_manager_lib 
    PUBLIC 
    ${LOCAL_INCL_DIR} ${EIGEN_INCL} ${TOML_INCL})
target_link_libraries(safety_manager_lib joint_filter_lib)
target_compile_options(safety_manager_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

add_library(motor_control_lib STATIC ${LOCAL_SRC_DIR}/motor_control.cpp)
target_include_directories(motor_control_lib PUBLIC ${LOCAL_INCL_DIR} ${EIGEN_INCL} ${TOML_INCL})
target_link_libraries(motor_control_lib safety_manager_lib joint_filter_lib)
target_compile_options(motor_control_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

add_library(math_expressions_lib STATIC ${EXPR_SRCS})
target_include_directories(math_expressions_lib PUBLIC ${LOCAL_INCL_DIR} ${EIGEN_INCL} ${TOML_INCL} ${EXPR_INCL})
target_link_libraries(math_expressions_lib  safety_manager_lib)
target_compile_options(math_expressions_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

add_library(robot_expressions_lib STATIC ${LOCAL_SRC_DIR}/robot_expressions.cpp ${EXPR_SRCS})
target_include_directories(robot_expressions_lib PUBLIC ${LOCAL_INCL_DIR} ${EIGEN_INCL} ${TOML_INCL} ${EXPR_INCL})
target_link_libraries(robot_expressions_lib safety_manager_lib math_expressions_lib)
target_compile_options(robot_expressions_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

add_library(walking_trajectory_lib STATIC ${LOCAL_SRC_DIR}/walking_trajectory.cpp ${LOCAL_SRC_DIR}/digit_state.cpp)
target_include_directories(walking_trajectory_lib PUBLIC ${LOCAL_INCL_DIR} ${EIGEN_INCL} ${TOML_INCL})
target_link_libraries(walking_trajectory_lib safety_manager_lib joint_filter_lib robot_expressions_lib kinematics_dynamics_lib)
target_compile_options(walking_trajectory_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

add_library(domain_control_lib STATIC ${LOCAL_SRC_DIR}/domain_control.cpp)
target_include_directories(domain_control_lib PUBLIC ${LOCAL_INCL_DIR} ${EIGEN_INCL})
target_compile_options(domain_control_lib PRIVATE -fPIC -O3 -flto -mno-avx2)

#Create main control program
add_library(digit_controller_lib ${LOCAL_SRC_DIR}/digit_controller.cpp)
# add_library(digit_controller_lib ${LOCAL_INCL_DIR}/digit_controller.hpp)
target_include_directories(digit_controller_lib PRIVATE  DigitControlPybind/src/Cutils)
target_link_libraries(digit_controller_lib
  safety_manager_lib
  motor_control_lib 
  walking_trajectory_lib 
  kinematics_dynamics_lib
  geometric_kinematics_lib
  domain_control_lib
)
target_compile_options(digit_controller_lib PRIVATE -fPIC -O3 -flto -mno-avx2)


#######################################
# # Pybind11
#######################################
pybind11_add_module(${PROJECT_NAME} DigitControlPybind/src/digit_pybind.cpp DigitControlPybind/src/Cutils/Yaml.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE digit_controller_lib)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS}
DigitControlPybind/src/include/digit_controller/include
DigitControlPybind/src/include
DigitControlPybind/src/Cutils)

target_compile_options(${PROJECT_NAME} PRIVATE -fPIC -O3 -flto -mno-avx2)

target_compile_definitions(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:EIGEN_NO_DEBUG>")
target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DDIGIT_CONTROL_ENV_NAME=${PROJECT_NAME}")
