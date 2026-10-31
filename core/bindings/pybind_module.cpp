// Author: Paulo R. A. Gilz
//
// OPTIONAL pybind11 module.
//
// This is NOT the binding the application uses. visimu talks to the core
// through the plain C ABI in aoe_c_api.h via ctypes, because that needs no
// third-party build dependency and works with any CPython -- which matters on
// machines where pybind11 cannot be installed.
//
// This target exists for callers who would rather have a native extension
// module (a notebook, an analysis pipeline). Build it with:
//
//     cmake -B build -DAOE_BUILD_PYBIND=ON
//     cmake --build build -j
//
// It was written against the pybind11 2.13 API but could not be compiled in the
// environment this project was developed in, since pybind11 was unreachable
// there. Treat it as unverified until you have built it once.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aoe/decimate.hpp"
#include "aoe/iso.hpp"
#include "aoe/session.hpp"

namespace py = pybind11;

namespace {

// Wraps a core-owned buffer as a NumPy array without copying. The `base` object
// keeps the owning Session alive for as long as the array exists, which is what
// stops a view outliving the memory behind it.
py::array_t<float> view_channel(aoe::Session& session, py::object owner, int file,
                                int sim, const std::string& name) {
    std::size_t count = 0;
    const float* data = session.channel(file, sim, name, &count);
    if (!data || count == 0) return py::array_t<float>(0);

    return py::array_t<float>({static_cast<py::ssize_t>(count)},
                              {static_cast<py::ssize_t>(sizeof(float))}, data, owner);
}

}  // namespace

PYBIND11_MODULE(aoecore_py, m) {
    m.doc() = "Native bindings for the visimu core (optional; ctypes is the default).";

    py::class_<aoe::IsoGroup>(m, "IsoGroup")
        .def(py::init<>())
        .def_readwrite("name", &aoe::IsoGroup::name)
        .def_readwrite("active", &aoe::IsoGroup::active)
        .def_readwrite("min_bound", &aoe::IsoGroup::min_bound)
        .def_readwrite("max_bound", &aoe::IsoGroup::max_bound);

    py::class_<aoe::Session>(m, "Session")
        .def(py::init<>())
        // GIL released around every call that can block on I/O or fan out to
        // the thread pool, so other Python threads keep running during a load.
        .def("add_file", &aoe::Session::add_file, py::arg("path"),
             py::call_guard<py::gil_scoped_release>())
        .def("remove_file", &aoe::Session::remove_file, py::arg("index"))
        .def("clear", &aoe::Session::clear)
        .def_property_readonly("file_count", &aoe::Session::file_count)
        .def("file_name",
             [](aoe::Session& s, int i) {
                 const aoe::Dataset* d = s.file(i);
                 return d ? d->name() : std::string();
             })
        .def("file_path",
             [](aoe::Session& s, int i) {
                 const aoe::Dataset* d = s.file(i);
                 return d ? d->path() : std::string();
             })
        .def("headers",
             [](aoe::Session& s, int i) {
                 const aoe::Dataset* d = s.file(i);
                 return d ? d->headers() : std::vector<std::string>();
             })
        .def("simulation_count",
             [](aoe::Session& s, int i) {
                 const aoe::Dataset* d = s.file(i);
                 return d ? d->simulation_count() : std::size_t{0};
             })
        .def("simulation_id",
             [](aoe::Session& s, int f, int i) {
                 const aoe::Dataset* d = s.file(f);
                 if (!d || i < 0 || i >= static_cast<int>(d->simulation_count()))
                     return std::string();
                 return d->simulation(static_cast<std::size_t>(i)).id;
             })
        .def_property_readonly("parameters", &aoe::Session::parameters)
        .def("set_alias", &aoe::Session::set_alias)
        .def("remove_alias", &aoe::Session::remove_alias)
        .def("resolve", &aoe::Session::resolve)
        .def("set_computed",
             [](aoe::Session& s, const std::string& name, const std::string& expr) {
                 std::string error;
                 if (!s.set_computed(name, expr, &error))
                     throw std::runtime_error(error);
             })
        .def("remove_computed", &aoe::Session::remove_computed)
        .def("is_computed", &aoe::Session::is_computed)
        .def("preload", &aoe::Session::preload, py::arg("file"), py::arg("sims"),
             py::arg("names"), py::call_guard<py::gil_scoped_release>())
        .def("channel",
             [](py::object self, int file, int sim, const std::string& name) {
                 auto& session = self.cast<aoe::Session&>();
                 return view_channel(session, self, file, sim, name);
             },
             py::arg("file"), py::arg("sim"), py::arg("name"))
        .def("iso_values", &aoe::Session::iso_values)
        .def("iso_distinct_values", &aoe::Session::iso_distinct_values);

    m.def("iso_auto_group", &aoe::iso_auto_group, py::arg("values"),
          py::arg("group_count"));
    m.def("iso_value_active", &aoe::iso_value_active);
    m.def("iso_group_name", &aoe::iso_group_name);
    m.def("iso_format_value", &aoe::iso_format_value);

    m.def(
        "decimate",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
           py::array_t<float, py::array::c_style | py::array::forcecast> y,
           std::size_t max_points, float x_min, float x_max) {
            const std::size_t count =
                std::min(static_cast<std::size_t>(x.size()),
                         static_cast<std::size_t>(y.size()));
            const std::size_t capacity = 2 * max_points + 4;

            py::array_t<float> out_x(static_cast<py::ssize_t>(capacity));
            py::array_t<float> out_y(static_cast<py::ssize_t>(capacity));

            std::size_t written = 0;
            {
                py::gil_scoped_release release;
                if (x_min < x_max)
                    written = aoe::decimate_minmax_range(
                        x.data(), y.data(), count, x_min, x_max, max_points,
                        out_x.mutable_data(), out_y.mutable_data());
                else
                    written = aoe::decimate_minmax(x.data(), y.data(), count, max_points,
                                                   out_x.mutable_data(),
                                                   out_y.mutable_data());
            }

            out_x.resize({static_cast<py::ssize_t>(written)});
            out_y.resize({static_cast<py::ssize_t>(written)});
            return py::make_tuple(out_x, out_y);
        },
        py::arg("x"), py::arg("y"), py::arg("max_points"), py::arg("x_min") = 0.0f,
        py::arg("x_max") = 0.0f);

    m.attr("__version__") = "2.0.0";
}
