
#include "dup.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dup_forward", &dup_forward, "dup_forward");
  m.def("dup_backward", &dup_backward, "dup_backward");
}
