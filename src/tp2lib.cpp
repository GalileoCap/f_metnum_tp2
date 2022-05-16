#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "pca.h"
#include "knn.h"

PYBIND11_MODULE(tp2, m) {
  py::class_<PCA>(m, "PCA")
    .def(py::init<uint, uint>())
    .def("fit", &PCA::fit, py::arg("M"))
    .def("transform", &PCA::transform, py::arg("X"))
    .def_readonly("components_", &PCA::_M);

  py::class_<KNN>(m, "KNN")
    .def(py::init<uint>())
    .def("fit", &KNN::fit, py::arg("X"), py::arg("Y"))
    .def("predict", &KNN::predict, py::arg("X"));
}

