#include <stdio.h>

#define BLOCK_SIZE 32
#define FULL_MASK 0xffffffff

__constant__ float filter[3 * 3];


__device__ void full_load(const float *input, float *s_input, int rows, int cols, int padding, unsigned total_padding, int tile_size, int tid_x, int tid_y) {
    int overall_x = tid_x - padding;
    int overall_y = tid_y - padding;
    unsigned wid = (threadIdx.y * BLOCK_SIZE + threadIdx.x) / warpSize;
    s_input[threadIdx.y * tile_size + threadIdx.x] = (overall_x >= 0 && overall_x < cols && overall_y >= 0 && overall_y < rows) ? input[overall_y * cols + overall_x] : 0.f;
   
    unsigned num_warps = total_padding + (tile_size + 2 - 1) / 2;  // ceiling division
    if (wid < total_padding) {
        overall_y += BLOCK_SIZE;
        if ((BLOCK_SIZE + threadIdx.y) < tile_size && threadIdx.x < tile_size) {
            s_input[(BLOCK_SIZE + threadIdx.y) * tile_size + threadIdx.x] = (overall_x >= 0 && overall_x < cols && overall_y >= 0 && overall_y < rows) ? \
            input[overall_y * cols + overall_x] : 0.f;
        }
    }
    else if (wid < num_warps) {
        unsigned shared_y = threadIdx.x / 16 + 2 * (wid - total_padding);
        unsigned shared_x = BLOCK_SIZE + threadIdx.x % total_padding;
        if (shared_x < tile_size && shared_y < tile_size) {
            overall_y = blockIdx.y * blockDim.y + shared_y - padding;
            overall_x = blockIdx.x * blockDim.x + shared_x - padding;
            s_input[shared_y * tile_size + shared_x] = \
            (overall_x >= 0 && overall_x < cols && overall_y >= 0 && overall_y < rows) ? input[overall_y * cols + overall_x] : 0.f;
        }
    }
}


__global__ void conv_2d_shfl(const float *input, float *out, int rows, int cols, int filter_size) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int padding = filter_size / 2;
    unsigned total_padding = padding * 2;
    int tile_size = BLOCK_SIZE + padding * 2;
    float val = 0.f;

    extern __shared__ float s_input[];

    // copy global input into shared input
    for (int tile_row = 0; tile_row < tile_size / BLOCK_SIZE; ++tile_row) {
            int tile_y = threadIdx.y + tile_row * BLOCK_SIZE;
            int overall_y = tid_y - padding + tile_row * BLOCK_SIZE;
        for (int tile_col = 0; tile_col < tile_size / BLOCK_SIZE; ++tile_col) {
            // three regions - inside array, outside array inside padding, outside padding
            int tile_x = threadIdx.x + tile_col * BLOCK_SIZE;
            if (tile_x >= tile_size || tile_y >= tile_size) continue;  // outside padding/(array where thread should read from)

            // everything else will be inside array or padding
            int overall_x = tid_x - padding + tile_col * BLOCK_SIZE;
            s_input[tile_y * tile_size + tile_x] = (overall_x >= 0 && overall_x < cols && overall_y >= 0 && overall_y < rows) ? input[overall_y * cols + overall_x] : 0.f;
        }
    }
    __syncthreads();

    bool inside_array = tid_x >= 0 && tid_x < cols && tid_y >= 0 && tid_y < rows;
    float2 next_vals;
    for (int fid_y = 0; fid_y < filter_size; ++fid_y) {
        int idx_y = threadIdx.y + fid_y;
        float arr_val = s_input[idx_y * tile_size + threadIdx.x];  // initial row load
        if (threadIdx.x == 31) next_vals = reinterpret_cast<float2*>(s_input)[32];  // vectorized load by last thread
        val += inside_array ? arr_val * filter[fid_y * filter_size] : 0.f;
        arr_val = __shfl_down_sync(FULL_MASK, arr_val, 1);  // loading from threads above
        next_vals.x = __shfl_down_sync(FULL_MASK, arr_val, 1);
        for (int fid_x = 1; fid_x < filter_size; ++fid_x) {
            val += inside_array ? arr_val * filter[fid_y * filter_size + fid_x] : 0.f;
            arr_val = next_vals.x;
        }
    }

    if (inside_array)
        out[tid_y * cols + tid_x] = val;
}


void fill_array(float *arr, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        arr[i] = 1.f;
    }
}


void print_matrix(float *arr, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.2f ", arr[i * cols + j]);
        }
        printf("\n");
    }
}


int main() {
    // Event for timing

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int rows = 2048, cols = 2048;
    int filter_size = 3;
    float *img, *h_filter, *out;
    float *d_img, *d_out;

    // allocate memory
    cudaMallocHost((void **)&img, rows * cols * sizeof(float));
    cudaMallocHost((void **)&h_filter, filter_size * filter_size * sizeof(float));
    out = (float *)malloc(rows * cols * sizeof(float));

    fill_array(img, rows, cols);
    fill_array(h_filter, filter_size, filter_size);

    cudaMalloc((void **)&d_img, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_out, rows * cols * sizeof(float));

    // copy initialized arrays to device memory
    cudaMemcpy(d_img, img, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(filter, h_filter, filter_size * filter_size * sizeof(float));

    //kernel call
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid((rows + BLOCK_SIZE - 1) / BLOCK_SIZE, (cols + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    int tile_size = BLOCK_SIZE + (filter_size / 2) * 2;
    conv_2d_shfl<<<dimGrid, dimBlock, tile_size * tile_size * sizeof(float), 0>>>(d_img, d_out, rows, cols, filter_size);
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ = 0.f;
    cudaEventElapsedTime(&time_, start, stop);
    printf("Shuffle elapsed time (ms): %.3f \n", time_);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(out, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    // print_matrix(out, rows, cols);

    // frees
    cudaFree(d_img);
    cudaFree(d_out);
    cudaFreeHost(img);
    cudaFreeHost(h_filter);
    free(out);

    return 0;
}

