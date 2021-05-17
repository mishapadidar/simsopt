#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "py_shared_ptr.h"
PYBIND11_DECLARE_HOLDER_TYPE(T, py_shared_ptr<T>);
using std::shared_ptr;
using std::vector;

#include "magneticfield.h"
#include "pymagneticfield.h"
#include "regular_grid_interpolant_3d.h"
typedef MagneticField<PyArray> PyMagneticField;
typedef BiotSavart<PyArray> PyBiotSavart;


template <typename T, typename S> void register_common_field_methods(S &c) {
    c.def("set_points", &T::set_points)
     .def("B", py::overload_cast<>(&T::B), py::call_guard<py::gil_scoped_release>())
     .def("dB_by_dX", py::overload_cast<>(&T::dB_by_dX), py::call_guard<py::gil_scoped_release>())
     .def("d2B_by_dXdX", py::overload_cast<>(&T::d2B_by_dXdX), py::call_guard<py::gil_scoped_release>())
     .def("B_ref", py::overload_cast<>(&T::B_ref), py::call_guard<py::gil_scoped_release>())
     .def("dB_by_dX_ref", py::overload_cast<>(&T::dB_by_dX_ref), py::call_guard<py::gil_scoped_release>())
     .def("d2B_by_dXdX_ref", py::overload_cast<>(&T::d2B_by_dXdX_ref), py::call_guard<py::gil_scoped_release>())
     .def("A", py::overload_cast<>(&T::A), py::call_guard<py::gil_scoped_release>())
     .def("dA_by_dX", py::overload_cast<>(&T::dA_by_dX), py::call_guard<py::gil_scoped_release>())
     .def("d2A_by_dXdX", py::overload_cast<>(&T::d2A_by_dXdX), py::call_guard<py::gil_scoped_release>())
     .def("A_ref", py::overload_cast<>(&T::A_ref), py::call_guard<py::gil_scoped_release>())
     .def("dA_by_dX_ref", py::overload_cast<>(&T::dA_by_dX_ref), py::call_guard<py::gil_scoped_release>())
     .def("d2A_by_dXdX_ref", py::overload_cast<>(&T::d2A_by_dXdX_ref), py::call_guard<py::gil_scoped_release>())
     .def("cache_get_or_create", &T::cache_get_or_create)
     .def("cache_get_status", &T::cache_get_status)
     .def("invalidate_cache", &T::invalidate_cache)
     .def_readwrite("points", &T::points);
}

void init_magneticfields(py::module_ &m){

    py::class_<RegularGridInterpolant3D<PyArray, 1>>(m, "RegularGridInterpolant3D1")
        .def(py::init<int, int, int, int>())
        .def(py::init<RangeTriplet, RangeTriplet, RangeTriplet, int>())
        .def("interpolate", &RegularGridInterpolant3D<PyArray, 1>::interpolate)
        .def("interpolate_batch", &RegularGridInterpolant3D<PyArray, 1>::interpolate_batch)
        .def("evaluate_batch_with_transform", &RegularGridInterpolant3D<PyArray, 1>::evaluate_batch_with_transform)
        .def("evaluate_batch", &RegularGridInterpolant3D<PyArray, 1>::evaluate_batch)
        .def("evaluate", &RegularGridInterpolant3D<PyArray, 1>::evaluate)
        .def("estimate_error", &RegularGridInterpolant3D<PyArray, 1>::estimate_error);
    py::class_<RegularGridInterpolant3D<PyArray, 4>>(m, "RegularGridInterpolant3D4")
        .def(py::init<int, int, int, int>())
        .def(py::init<RangeTriplet, RangeTriplet, RangeTriplet, int>())
        .def("interpolate", &RegularGridInterpolant3D<PyArray, 4>::interpolate)
        .def("interpolate_batch", &RegularGridInterpolant3D<PyArray, 4>::interpolate_batch)
        .def("evaluate_batch_with_transform", &RegularGridInterpolant3D<PyArray, 4>::evaluate_batch_with_transform)
        .def("evaluate_batch", &RegularGridInterpolant3D<PyArray, 4>::evaluate_batch)
        .def("evaluate", &RegularGridInterpolant3D<PyArray, 4>::evaluate)
        .def("estimate_error", &RegularGridInterpolant3D<PyArray, 4>::estimate_error);

    py::class_<Current<PyArray>, shared_ptr<Current<PyArray>>>(m, "Current")
        .def(py::init<double>())
        .def("set_dofs", &Current<PyArray>::set_dofs)
        .def("get_dofs", &Current<PyArray>::get_dofs)
        .def("set_value", &Current<PyArray>::set_value)
        .def("get_value", &Current<PyArray>::get_value);
        

    py::class_<Coil<PyArray>, shared_ptr<Coil<PyArray>>>(m, "Coil")
        .def(py::init<shared_ptr<Curve<PyArray>>, shared_ptr<Current<PyArray>>>())
        .def_readonly("curve", &Coil<PyArray>::curve)
        .def_readonly("current", &Coil<PyArray>::current);

    auto mf = py::class_<PyMagneticField, PyMagneticFieldTrampoline<PyMagneticField>>(m, "MagneticField")
        .def(py::init<>());
    register_common_field_methods<PyMagneticField>(mf);
        //.def("B", py::overload_cast<>(&PyMagneticField::B));

    auto bs = py::class_<PyBiotSavart, PyMagneticFieldTrampoline<PyBiotSavart>, PyMagneticField>(m, "BiotSavart")
        .def(py::init<vector<shared_ptr<Coil<PyArray>>>>())
        .def("compute", &PyBiotSavart::compute);
    register_common_field_methods<PyBiotSavart>(bs);
}