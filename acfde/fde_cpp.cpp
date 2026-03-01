#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

namespace py = pybind11;

void compute_fde_query_batch(
    const float* points,
    const int64_t* offsets,
    const float* simhash_matrices,
    const float* ams_matrices,
    float* output,
    int num_queries,
    int total_vectors,
    int original_dim,
    int keep_R,
    int k,
    int proj_dim,
    int seed
) {
    int num_partitions = 1 << k;
    int final_fde_dim = keep_R * num_partitions * proj_dim;
    
    #pragma omp parallel
    {
        std::vector<float> sketches(k);
        std::vector<float> projected(proj_dim);
        std::vector<float> rep_fde;
        
        #pragma omp for schedule(dynamic)
        for (int q = 0; q < num_queries; q++) {
            int s = offsets[q];
            int e = offsets[q + 1];
            int num_points = e - s;
            
            if (num_points == 0) continue;
            
            for (int r = 0; r < keep_R; r++) {
                rep_fde.assign(num_partitions * proj_dim, 0.0f);
                
                const float* simhash_mat = simhash_matrices + r * original_dim * k;
                const float* ams_mat = (ams_matrices != nullptr) ? ams_matrices + r * original_dim * proj_dim : nullptr;
                
                for (int i = s; i < e; i++) {
                    const float* pt = points + i * original_dim;
                    
                    for (int j = 0; j < k; j++) {
                        float val = 0.0f;
                        for (int d = 0; d < original_dim; d++) {
                            val += pt[d] * simhash_mat[d * k + j];
                        }
                        sketches[j] = val;
                    }
                    
                    if (ams_mat != nullptr) {
                        for (int j = 0; j < proj_dim; j++) {
                            float val = 0.0f;
                            for (int d = 0; d < original_dim; d++) {
                                val += pt[d] * ams_mat[d * proj_dim + j];
                            }
                            projected[j] = val;
                        }
                    } else {
                        for (int j = 0; j < proj_dim; j++) {
                            projected[j] = pt[j];
                        }
                    }
                    
                    int part_idx = 0;
                    if (k > 0) {
                        for (int bit_idx = 0; bit_idx < k; bit_idx++) {
                            int bit = (sketches[bit_idx] > 0) ? 1 : 0;
                            part_idx = (part_idx << 1) + (bit ^ (part_idx & 1));
                        }
                    }
                    
                    int base = part_idx * proj_dim;
                    for (int j = 0; j < proj_dim; j++) {
                        rep_fde[base + j] += projected[j];
                    }
                }
                
                int rep_start = r * num_partitions * proj_dim;
                for (int j = 0; j < num_partitions * proj_dim; j++) {
                    output[q * final_fde_dim + rep_start + j] = rep_fde[j];
                }
            }
        }
    }
}

py::array_t<float> fde_query_cpp(
    py::array_t<float> points_np,
    py::array_t<int64_t> offsets_np,
    py::array_t<float> simhash_stack_np,
    py::object ams_stack_np,
    int num_queries,
    int total_vectors,
    int original_dim,
    int keep_R,
    int k,
    int proj_dim
) {
    auto points = points_np.unchecked<2>();
    auto offsets = offsets_np.unchecked<1>();
    auto simhash_stack = simhash_stack_np.unchecked<3>();
    
    int num_partitions = 1 << k;
    int final_fde_dim = keep_R * num_partitions * proj_dim;
    
    py::array_t<float> output_np({num_queries, final_fde_dim});
    auto output = output_np.mutable_unchecked<2>();
    
    const float* ams_ptr = nullptr;
    py::array_t<float> ams_arr;
    if (!ams_stack_np.is_none()) {
        ams_arr = ams_stack_np.cast<py::array_t<float>>();
        ams_ptr = ams_arr.unchecked<3>().data(0, 0, 0);
    }
    
    compute_fde_query_batch(
        points.data(0, 0),
        offsets.data(0),
        simhash_stack.data(0, 0, 0),
        ams_ptr,
        output.mutable_data(0, 0),
        num_queries,
        total_vectors,
        original_dim,
        keep_R,
        k,
        proj_dim,
        42
    );
    
    return output_np;
}

PYBIND11_MODULE(fde_cpp, m) {
    m.doc() = "FDE generation C++ extension";
    m.def("fde_query_cpp", &fde_query_cpp, 
          "Generate FDE for queries using C++",
          py::arg("points"),
          py::arg("offsets"),
          py::arg("simhash_stack"),
          py::arg("ams_stack"),
          py::arg("num_queries"),
          py::arg("total_vectors"),
          py::arg("original_dim"),
          py::arg("keep_R"),
          py::arg("k"),
          py::arg("proj_dim"));
}
