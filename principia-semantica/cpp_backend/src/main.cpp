// cpp_backend/src/main.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

// A simple function to prove the build system works.
int add(int i, int j) {
    return i + j;
}

// Module definition
// The first argument is the name of the module in Python.
// The second argument is a handle to the module object.
PYBIND11_MODULE(cpp_backend, m) {
    m.doc() = "High-performance C++ backend for Principia Semantica";

    m.def("add", &add, "A function that adds two numbers",
          py::arg("i"), py::arg("j"));
    
    // Placeholder for future high-performance functions
    // m.def("fast_knn_graph", &fast_knn_graph, "Constructs k-NN graph using Faiss");
}