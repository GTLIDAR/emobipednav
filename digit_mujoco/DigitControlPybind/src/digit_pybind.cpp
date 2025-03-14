#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "digitControlEnv.hpp"

namespace py = pybind11;

PYBIND11_MODULE(DIGIT_CONTROL_ENV_NAME, m) {
  py::class_<digitControlEnv>(m, "digitControlEnv")
      .def(py::init<double>())
      .def("init", &digitControlEnv::init)
      .def("reset", &digitControlEnv::reset)
      .def("setStates", &digitControlEnv::setStates)
      .def("computeTorque", &digitControlEnv::computeTorque)
      .def("setUsrCommand", &digitControlEnv::setUsrCommand)
      .def("getPhaseVariable", &digitControlEnv::getPhaseVariable)
      .def("getDomain", &digitControlEnv::getDomain)
      .def("getDomainSwitch", &digitControlEnv::getDomainSwitch);
}
