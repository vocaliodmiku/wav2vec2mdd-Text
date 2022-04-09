#include "viterbi.h"
#include <omp.h> 
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>

namespace Simpleviterbi {
template <class Float>
size_t ViterbiPath<Float>::getWorkspaceSize(int B, int T, int N) {
    return WorkspacePtrs<Float>(nullptr, B, T, N).requiredSize;
}

template <class Float>
void ViterbiPath<Float>::compute(
    int B,
    int T,
    int N,
    const Float* input,
    const Float* trans,
    int* _path,
    void* workspace) {
    WorkspacePtrs<Float> ws(workspace, B, T, N);
#pragma omp parallel for num_threads(B)
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            ws.alpha[b * 2 * N + n] = input[b * T * N + n];
        }
        for (int t = 1; t <= T; ++t) {
            const auto* alphaPrev = &ws.alpha[b * 2 * N + ((t - 1) % 2) * N];
            const auto* inputCur = &input[b * T * N + t * N];
            auto* alphaCur = &ws.alpha[b * 2 * N + (t % 2) * N];
            auto* betaCur = &ws.beta[b * T * N + t * N];

            for (int m = 0; m < N; ++m) {
                int maxIndex = -1;
                Float maxValue = -INFINITY;
                for (int n = 0; n < N; ++n) {
                    Float val = alphaPrev[n] + (t == T ? 0 : trans[m * N + n]);
                    if (val > maxValue) {
                        maxIndex = n;
                        maxValue = val;
                    }
                }

                if (t == T) {
                    auto* path = &_path[b * T];
                    path[T - 1] = maxIndex;
                    for (int s = T - 1; s > 0; --s) {
                        path[s - 1] = ws.beta[b * T * N + s * N + path[s]];
                    }
                    break;
                }

                alphaCur[m] = maxValue + inputCur[m];
                betaCur[m] = maxIndex;
            }
        }
    }
    }
}
namespace py = pybind11;

template <class T>
static T castBytes(const py::bytes& b) {
  static_assert(
      std::is_standard_layout<T>::value,
      "types represented as bytes must be standard layout");
  std::string s = b;
  if (s.size() != sizeof(T)) {
    throw std::runtime_error("wrong py::bytes size to represent object");
  }
  return *reinterpret_cast<const T*>(s.data());
}

static void CpuViterbi_compute(
    int B,
    int T,
    int N,
    py::bytes input,
    py::bytes trans,
    py::bytes path,
    py::bytes workspace) {
  Simpleviterbi::ViterbiPath<float>::compute(
      B,
      T,
      N,
      castBytes<const float*>(input),
      castBytes<const float*>(trans),
      castBytes<int*>(path),
      castBytes<void*>(workspace));
}

PYBIND11_MODULE(viterbi, m) {
    m.doc() = "aaa";
    m.def("compute", &CpuViterbi_compute);
    m.def("get_workspace_size", &Simpleviterbi::ViterbiPath<float>::getWorkspaceSize);
}
// int main(int argc, char* argv[]){
//     int B = 1, T = 3, N = 3;
//     std::vector<float> data = {0.25, 0.40, 0.35, 0.40, 0.35, 0.25, 0.10, 0.50, 0.40};
//     std::vector<float> exp_data;
//     exp_data.reserve(data.size());
//     for (size_t i = 0; i < data.size(); ++i) {
//         exp_data.push_back(std::log(data[i]));
//     }
    
//     torch::Tensor emissions =  torch::from_blob(exp_data.data(), {B, T, N}, torch::kFloat);
//     torch::Tensor transitions = torch::zeros({N, N}, torch::kFloat);
//     torch::Tensor viterbi_path = torch::zeros({B, T}, torch::kInt);
//     torch::Tensor workspace = torch::zeros(myviterbi::ViterbiPath<float>::getWorkspaceSize(B, T, N), torch::kByte);

//     myviterbi::ViterbiPath<float>::compute(
//         B,
//         T,
//         N,
//         reinterpret_cast<const float*>(emissions.data_ptr()),
//         reinterpret_cast<const float*>(transitions.data_ptr()),
//         reinterpret_cast<int*>(viterbi_path.data_ptr()),
//         workspace.data_ptr()
//     );
//     for (int i = 0; i < T; ++i) {
//         std::cout << viterbi_path[0][i].item<int>();
//         std::cout << " ";
//     };
//     return 0;
// }
