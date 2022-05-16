#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "pca.h"

PYBIND11_MODULE(tp2, m) {
  py::class_<PCA>(m, "PCA")
    .def(py::init<uint, uint>())
    .def("fit", &PCA::fit, py::arg("M"))
    .def("transform", &PCA::transform, py::arg("X"))
    .def_readonly("components_", &PCA::_M);
}
