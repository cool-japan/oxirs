//! CUDA kernel implementations for various vector operations

// Note: anyhow imports removed as they were unused
use std::collections::HashMap;

/// CUDA kernel manager
#[derive(Debug)]
pub struct KernelManager {
    kernels: HashMap<String, String>,
}

impl KernelManager {
    pub fn new() -> Self {
        let mut manager = Self {
            kernels: HashMap::new(),
        };
        manager.initialize_kernels();
        manager
    }

    fn initialize_kernels(&mut self) {
        let kernels = vec![
            // Similarity metrics
            (
                "cosine_similarity".to_string(),
                self.get_cosine_similarity_kernel(),
            ),
            ("dot_product".to_string(), self.get_dot_product_kernel()),
            (
                "pearson_correlation".to_string(),
                self.get_pearson_correlation_kernel(),
            ),
            (
                "jaccard_similarity".to_string(),
                self.get_jaccard_similarity_kernel(),
            ),
            (
                "dice_coefficient".to_string(),
                self.get_dice_coefficient_kernel(),
            ),
            (
                "angular_similarity".to_string(),
                self.get_angular_similarity_kernel(),
            ),
            // Distance metrics
            (
                "euclidean_distance".to_string(),
                self.get_euclidean_distance_kernel(),
            ),
            (
                "manhattan_distance".to_string(),
                self.get_manhattan_distance_kernel(),
            ),
            (
                "minkowski_distance".to_string(),
                self.get_minkowski_distance_kernel(),
            ),
            (
                "hamming_distance".to_string(),
                self.get_hamming_distance_kernel(),
            ),
            (
                "canberra_distance".to_string(),
                self.get_canberra_distance_kernel(),
            ),
            (
                "chebyshev_distance".to_string(),
                self.get_chebyshev_distance_kernel(),
            ),
            // Utility kernels
            (
                "vector_addition".to_string(),
                self.get_vector_addition_kernel(),
            ),
            (
                "vector_normalization".to_string(),
                self.get_vector_normalization_kernel(),
            ),
            ("hnsw_search".to_string(), self.get_hnsw_search_kernel()),
            (
                "batch_distance_computation".to_string(),
                self.get_batch_distance_kernel(),
            ),
            // Mixed-precision kernels (FP16/BF16)
            (
                "cosine_similarity_fp16".to_string(),
                self.get_cosine_similarity_fp16_kernel(),
            ),
            (
                "euclidean_distance_fp16".to_string(),
                self.get_euclidean_distance_fp16_kernel(),
            ),
            // Tensor Core kernels
            (
                "matmul_tensor_core".to_string(),
                self.get_matmul_tensor_core_kernel(),
            ),
        ];

        for (name, kernel) in kernels {
            self.kernels.insert(name, kernel);
        }
    }

    pub fn get_kernel(&self, name: &str) -> Option<&String> {
        self.kernels.get(name)
    }

    fn get_cosine_similarity_kernel(&self) -> String {
        r#"
        extern "C" __global__ void cosine_similarity_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            float dot = 0.0f, norm_q = 0.0f, norm_db = 0.0f;

            const int vec_dim = (dim + 3) / 4;
            const float4* q_vec = (const float4*)(queries + query_idx * dim);
            const float4* db_vec = (const float4*)(database + db_idx * dim);

            for (int i = 0; i < vec_dim; i++) {
                float4 q_vals = q_vec[i];
                float4 db_vals = db_vec[i];

                dot += q_vals.x * db_vals.x + q_vals.y * db_vals.y +
                       q_vals.z * db_vals.z + q_vals.w * db_vals.w;
                norm_q += q_vals.x * q_vals.x + q_vals.y * q_vals.y +
                          q_vals.z * q_vals.z + q_vals.w * q_vals.w;
                norm_db += db_vals.x * db_vals.x + db_vals.y * db_vals.y +
                           db_vals.z * db_vals.z + db_vals.w * db_vals.w;
            }

            const float norm_product = sqrtf(norm_q) * sqrtf(norm_db);
            const float similarity = (norm_product > 1e-8f) ? dot / norm_product : 0.0f;

            results[query_idx * db_count + db_idx] = similarity;
        }
        "#
        .to_string()
    }

    fn get_euclidean_distance_kernel(&self) -> String {
        r#"
        extern "C" __global__ void euclidean_distance_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            float sum_sq_diff = 0.0f;

            const int vec_dim = (dim + 3) / 4;
            const float4* q_vec = (const float4*)(queries + query_idx * dim);
            const float4* db_vec = (const float4*)(database + db_idx * dim);

            for (int i = 0; i < vec_dim; i++) {
                float4 q_vals = q_vec[i];
                float4 db_vals = db_vec[i];
                float4 diff = make_float4(
                    q_vals.x - db_vals.x,
                    q_vals.y - db_vals.y,
                    q_vals.z - db_vals.z,
                    q_vals.w - db_vals.w
                );
                sum_sq_diff += diff.x * diff.x + diff.y * diff.y +
                               diff.z * diff.z + diff.w * diff.w;
            }

            results[query_idx * db_count + db_idx] = sqrtf(sum_sq_diff);
        }
        "#
        .to_string()
    }

    fn get_dot_product_kernel(&self) -> String {
        r#"
        extern "C" __global__ void dot_product_kernel(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ result,
            const int n
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;

            float sum = 0.0f;
            for (int i = tid; i < n; i += stride) {
                sum += a[i] * b[i];
            }

            __shared__ float shared_sum[256];
            shared_sum[threadIdx.x] = sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                atomicAdd(result, shared_sum[0]);
            }
        }
        "#
        .to_string()
    }

    fn get_vector_addition_kernel(&self) -> String {
        r#"
        extern "C" __global__ void vector_addition_kernel(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ result,
            const int n
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < n) {
                result[tid] = a[tid] + b[tid];
            }
        }
        "#
        .to_string()
    }

    fn get_vector_normalization_kernel(&self) -> String {
        r#"
        extern "C" __global__ void vector_normalization_kernel(
            float* __restrict__ vectors,
            const int count,
            const int dim
        ) {
            const int vector_idx = blockIdx.x;
            const int tid = threadIdx.x;

            if (vector_idx >= count) return;

            float* vector = vectors + vector_idx * dim;

            __shared__ float shared_norm;
            if (tid == 0) shared_norm = 0.0f;
            __syncthreads();

            float local_sum = 0.0f;
            for (int i = tid; i < dim; i += blockDim.x) {
                local_sum += vector[i] * vector[i];
            }

            atomicAdd(&shared_norm, local_sum);
            __syncthreads();

            if (tid == 0) {
                shared_norm = sqrtf(shared_norm);
                if (shared_norm > 1e-8f) shared_norm = 1.0f / shared_norm;
            }
            __syncthreads();

            for (int i = tid; i < dim; i += blockDim.x) {
                vector[i] *= shared_norm;
            }
        }
        "#
        .to_string()
    }

    fn get_hnsw_search_kernel(&self) -> String {
        r#"
        extern "C" __global__ void hnsw_search_kernel(
            const float* __restrict__ query,
            const float* __restrict__ vectors,
            const int* __restrict__ adjacency_list,
            const int* __restrict__ adjacency_offsets,
            int* __restrict__ candidate_queue,
            float* __restrict__ candidate_distances,
            int* __restrict__ queue_size,
            const int dim,
            const int entry_point
        ) {
            const int tid = threadIdx.x;

            extern __shared__ float shared_data[];
            float* shared_query = shared_data;
            int* shared_queue = (int*)(shared_data + dim);
            float* shared_queue_dist = (float*)(shared_queue + 128);

            if (tid < dim) {
                shared_query[tid] = query[tid];
            }
            __syncthreads();

            int queue_head = 0;
            int queue_tail = 0;

            if (tid == 0) {
                shared_queue[0] = entry_point;
                shared_queue_dist[0] = 0.0f;
                queue_tail = 1;
            }
            __syncthreads();

            while (queue_head < queue_tail && queue_tail < 128) {
                __syncthreads();

                if (tid == 0 && queue_head < queue_tail) {
                    int current_node = shared_queue[queue_head];
                    queue_head++;

                    int neighbor_start = adjacency_offsets[current_node];
                    int neighbor_end = adjacency_offsets[current_node + 1];

                    for (int i = neighbor_start; i < neighbor_end && queue_tail < 128; i++) {
                        int neighbor = adjacency_list[i];

                        const float* neighbor_vector = vectors + neighbor * dim;
                        float neighbor_dist = 0.0f;
                        for (int d = 0; d < dim; d++) {
                            float diff = shared_query[d] - neighbor_vector[d];
                            neighbor_dist += diff * diff;
                        }
                        neighbor_dist = sqrtf(neighbor_dist);

                        shared_queue[queue_tail] = neighbor;
                        shared_queue_dist[queue_tail] = neighbor_dist;
                        queue_tail++;
                    }
                }
            }

            if (tid < queue_tail) {
                candidate_queue[tid] = shared_queue[tid];
                candidate_distances[tid] = shared_queue_dist[tid];
            }

            if (tid == 0) {
                *queue_size = queue_tail;
            }
        }
        "#
        .to_string()
    }

    fn get_batch_distance_kernel(&self) -> String {
        r#"
        extern "C" __global__ void batch_distance_kernel(
            const float* __restrict__ batch_a,
            const float* __restrict__ batch_b,
            float* __restrict__ distances,
            const int batch_size_a,
            const int batch_size_b,
            const int dim,
            const int metric_type
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int i = tid / batch_size_b;
            const int j = tid % batch_size_b;

            if (i >= batch_size_a || j >= batch_size_b) return;

            const float* vec_a = batch_a + i * dim;
            const float* vec_b = batch_b + j * dim;

            float distance = 0.0f;

            if (metric_type == 0) { // Euclidean
                for (int d = 0; d < dim; d++) {
                    float diff = vec_a[d] - vec_b[d];
                    distance += diff * diff;
                }
                distance = sqrtf(distance);
            } else if (metric_type == 1) { // Cosine
                float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
                for (int d = 0; d < dim; d++) {
                    dot += vec_a[d] * vec_b[d];
                    norm_a += vec_a[d] * vec_a[d];
                    norm_b += vec_b[d] * vec_b[d];
                }
                float norm_product = sqrtf(norm_a) * sqrtf(norm_b);
                distance = (norm_product > 1e-8f) ? 1.0f - (dot / norm_product) : 1.0f;
            }

            distances[i * batch_size_b + j] = distance;
        }
        "#
        .to_string()
    }

    // Additional distance metric kernels

    fn get_manhattan_distance_kernel(&self) -> String {
        r#"
        extern "C" __global__ void manhattan_distance_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            float sum_abs_diff = 0.0f;
            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            for (int i = 0; i < dim; i++) {
                sum_abs_diff += fabsf(q_vec[i] - db_vec[i]);
            }

            results[query_idx * db_count + db_idx] = sum_abs_diff;
        }
        "#
        .to_string()
    }

    fn get_minkowski_distance_kernel(&self) -> String {
        r#"
        extern "C" __global__ void minkowski_distance_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim,
            const float p
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            float sum_pow_diff = 0.0f;
            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            for (int i = 0; i < dim; i++) {
                float diff = fabsf(q_vec[i] - db_vec[i]);
                sum_pow_diff += powf(diff, p);
            }

            results[query_idx * db_count + db_idx] = powf(sum_pow_diff, 1.0f / p);
        }
        "#
        .to_string()
    }

    fn get_pearson_correlation_kernel(&self) -> String {
        r#"
        extern "C" __global__ void pearson_correlation_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            // Calculate means
            float mean_q = 0.0f, mean_db = 0.0f;
            for (int i = 0; i < dim; i++) {
                mean_q += q_vec[i];
                mean_db += db_vec[i];
            }
            mean_q /= dim;
            mean_db /= dim;

            // Calculate correlation
            float numerator = 0.0f, var_q = 0.0f, var_db = 0.0f;
            for (int i = 0; i < dim; i++) {
                float q_centered = q_vec[i] - mean_q;
                float db_centered = db_vec[i] - mean_db;
                numerator += q_centered * db_centered;
                var_q += q_centered * q_centered;
                var_db += db_centered * db_centered;
            }

            float denominator = sqrtf(var_q) * sqrtf(var_db);
            float correlation = (denominator > 1e-8f) ? numerator / denominator : 0.0f;

            results[query_idx * db_count + db_idx] = (correlation + 1.0f) / 2.0f;  // Normalize to [0, 1]
        }
        "#
        .to_string()
    }

    fn get_jaccard_similarity_kernel(&self) -> String {
        r#"
        extern "C" __global__ void jaccard_similarity_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            float intersection = 0.0f, union_val = 0.0f;
            for (int i = 0; i < dim; i++) {
                float min_val = fminf(q_vec[i], db_vec[i]);
                float max_val = fmaxf(q_vec[i], db_vec[i]);
                intersection += min_val;
                union_val += max_val;
            }

            float similarity = (union_val > 1e-8f) ? intersection / union_val : 0.0f;
            results[query_idx * db_count + db_idx] = similarity;
        }
        "#
        .to_string()
    }

    fn get_dice_coefficient_kernel(&self) -> String {
        r#"
        extern "C" __global__ void dice_coefficient_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            float intersection = 0.0f, sum_q = 0.0f, sum_db = 0.0f;
            for (int i = 0; i < dim; i++) {
                intersection += fminf(q_vec[i], db_vec[i]);
                sum_q += q_vec[i];
                sum_db += db_vec[i];
            }

            float denominator = sum_q + sum_db;
            float dice = (denominator > 1e-8f) ? (2.0f * intersection) / denominator : 0.0f;
            results[query_idx * db_count + db_idx] = dice;
        }
        "#
        .to_string()
    }

    fn get_hamming_distance_kernel(&self) -> String {
        r#"
        extern "C" __global__ void hamming_distance_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            int hamming_dist = 0;
            for (int i = 0; i < dim; i++) {
                if (fabsf(q_vec[i] - db_vec[i]) > 1e-6f) {
                    hamming_dist++;
                }
            }

            results[query_idx * db_count + db_idx] = (float)hamming_dist / (float)dim;
        }
        "#
        .to_string()
    }

    fn get_canberra_distance_kernel(&self) -> String {
        r#"
        extern "C" __global__ void canberra_distance_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            float distance = 0.0f;
            for (int i = 0; i < dim; i++) {
                float numerator = fabsf(q_vec[i] - db_vec[i]);
                float denominator = fabsf(q_vec[i]) + fabsf(db_vec[i]);
                if (denominator > 1e-8f) {
                    distance += numerator / denominator;
                }
            }

            results[query_idx * db_count + db_idx] = distance;
        }
        "#
        .to_string()
    }

    fn get_chebyshev_distance_kernel(&self) -> String {
        r#"
        extern "C" __global__ void chebyshev_distance_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            float max_diff = 0.0f;
            for (int i = 0; i < dim; i++) {
                float diff = fabsf(q_vec[i] - db_vec[i]);
                max_diff = fmaxf(max_diff, diff);
            }

            results[query_idx * db_count + db_idx] = max_diff;
        }
        "#
        .to_string()
    }

    fn get_angular_similarity_kernel(&self) -> String {
        r#"
        extern "C" __global__ void angular_similarity_kernel(
            const float* __restrict__ queries,
            const float* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            float dot = 0.0f, norm_q = 0.0f, norm_db = 0.0f;
            const float* q_vec = queries + query_idx * dim;
            const float* db_vec = database + db_idx * dim;

            for (int i = 0; i < dim; i++) {
                dot += q_vec[i] * db_vec[i];
                norm_q += q_vec[i] * q_vec[i];
                norm_db += db_vec[i] * db_vec[i];
            }

            float norm_product = sqrtf(norm_q) * sqrtf(norm_db);
            float cosine = (norm_product > 1e-8f) ? dot / norm_product : 0.0f;
            cosine = fminf(1.0f, fmaxf(-1.0f, cosine));  // Clamp to [-1, 1]

            // Angular distance in radians, normalized to [0, 1]
            float angular_dist = acosf(cosine) / 3.14159265359f;
            float similarity = 1.0f - angular_dist;

            results[query_idx * db_count + db_idx] = similarity;
        }
        "#
        .to_string()
    }

    // Mixed-precision kernels (FP16)

    fn get_cosine_similarity_fp16_kernel(&self) -> String {
        r#"
        #include <cuda_fp16.h>

        extern "C" __global__ void cosine_similarity_fp16_kernel(
            const half* __restrict__ queries,
            const half* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            float dot = 0.0f, norm_q = 0.0f, norm_db = 0.0f;
            const half* q_vec = queries + query_idx * dim;
            const half* db_vec = database + db_idx * dim;

            // Process in chunks of 2 for half2 vectorization
            const int vec_dim = dim / 2;
            const half2* q_vec2 = (const half2*)q_vec;
            const half2* db_vec2 = (const half2*)db_vec;

            for (int i = 0; i < vec_dim; i++) {
                half2 q_vals = q_vec2[i];
                half2 db_vals = db_vec2[i];

                float2 q_f = __half22float2(q_vals);
                float2 db_f = __half22float2(db_vals);

                dot += q_f.x * db_f.x + q_f.y * db_f.y;
                norm_q += q_f.x * q_f.x + q_f.y * q_f.y;
                norm_db += db_f.x * db_f.x + db_f.y * db_f.y;
            }

            // Handle odd dimension
            if (dim % 2 == 1) {
                float q_last = __half2float(q_vec[dim - 1]);
                float db_last = __half2float(db_vec[dim - 1]);
                dot += q_last * db_last;
                norm_q += q_last * q_last;
                norm_db += db_last * db_last;
            }

            const float norm_product = sqrtf(norm_q) * sqrtf(norm_db);
            const float similarity = (norm_product > 1e-8f) ? dot / norm_product : 0.0f;

            results[query_idx * db_count + db_idx] = similarity;
        }
        "#
        .to_string()
    }

    fn get_euclidean_distance_fp16_kernel(&self) -> String {
        r#"
        #include <cuda_fp16.h>

        extern "C" __global__ void euclidean_distance_fp16_kernel(
            const half* __restrict__ queries,
            const half* __restrict__ database,
            float* __restrict__ results,
            const int query_count,
            const int db_count,
            const int dim
        ) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int query_idx = tid / db_count;
            const int db_idx = tid % db_count;

            if (query_idx >= query_count || db_idx >= db_count) return;

            float sum_sq_diff = 0.0f;
            const half* q_vec = queries + query_idx * dim;
            const half* db_vec = database + db_idx * dim;

            const int vec_dim = dim / 2;
            const half2* q_vec2 = (const half2*)q_vec;
            const half2* db_vec2 = (const half2*)db_vec;

            for (int i = 0; i < vec_dim; i++) {
                float2 q_f = __half22float2(q_vec2[i]);
                float2 db_f = __half22float2(db_vec2[i]);

                float diff_x = q_f.x - db_f.x;
                float diff_y = q_f.y - db_f.y;
                sum_sq_diff += diff_x * diff_x + diff_y * diff_y;
            }

            if (dim % 2 == 1) {
                float diff = __half2float(q_vec[dim - 1]) - __half2float(db_vec[dim - 1]);
                sum_sq_diff += diff * diff;
            }

            results[query_idx * db_count + db_idx] = sqrtf(sum_sq_diff);
        }
        "#
        .to_string()
    }

    // Tensor Core kernel for matrix multiplication

    fn get_matmul_tensor_core_kernel(&self) -> String {
        r#"
        #include <mma.h>
        using namespace nvcuda;

        extern "C" __global__ void matmul_tensor_core_kernel(
            const half* __restrict__ a,
            const half* __restrict__ b,
            float* __restrict__ c,
            const int m,
            const int n,
            const int k
        ) {
            // Warp and lane identifiers
            const int warp_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;

            // WMMA fragment declarations (16x16x16)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

            // Initialize accumulator
            wmma::fill_fragment(c_frag, 0.0f);

            // Tile indices
            const int tile_m = blockIdx.y * 16;
            const int tile_n = blockIdx.x * 16;

            // Compute matrix multiplication using Tensor Cores
            for (int i = 0; i < k; i += 16) {
                wmma::load_matrix_sync(a_frag, a + tile_m * k + i, k);
                wmma::load_matrix_sync(b_frag, b + i * n + tile_n, n);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // Store result
            wmma::store_matrix_sync(c + tile_m * n + tile_n, c_frag, n, wmma::mem_row_major);
        }
        "#
        .to_string()
    }
}

impl Default for KernelManager {
    fn default() -> Self {
        Self::new()
    }
}
