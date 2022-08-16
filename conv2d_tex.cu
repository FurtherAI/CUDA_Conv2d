#include <stdio.h>

#define BLOCK_SIZE 32
#define MAX_FILTER 8
#define FULL_MASK 0xffffffff

__constant__ float filter[MAX_FILTER * MAX_FILTER];


__global__ void conv_2d_tex(cudaTextureObject_t texObj, float *out, int rows, int cols, int filter_size) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int padding = filter_size / 2;

    float val = 0.f;
    for (int fid_y = 0; fid_y < filter_size; ++fid_y) {
        for (int fid_x = 0; fid_x < filter_size; ++fid_x) {
            int idx_x = tid_x - padding + fid_x;
            int idx_y = tid_y - padding + fid_y;
            val += (tid_x >= 0 && tid_x < cols && tid_y >= 0 && tid_y < rows) ? tex2D<float>(texObj, (float)idx_x, (float)idx_y) * filter[fid_y * filter_size + fid_x] : 0.f;
        }
    }

    if (tid_x >= 0 && tid_x < cols && tid_y >= 0 && tid_y < rows)
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
    float *img, *h_filter, *out, *d_out;

    // allocate cudaArray (on devic) which holds the image
    cudaArray_t d_img;
    cudaChannelFormatDesc c_format = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&d_img, &c_format, cols, rows);

    // allocate memory for host image, host filter, host output and device output
    cudaMallocHost((void **)&img, rows * cols * sizeof(float));
    cudaMallocHost((void **)&h_filter, filter_size * filter_size * sizeof(float));
    out = (float *)malloc(rows * cols * sizeof(float));
    cudaMalloc((void **)&d_out, rows * cols * sizeof(float));

    // fill arrays with ones (easy to check for correct output)
    fill_array(img, rows, cols);
    fill_array(h_filter, filter_size, filter_size);

    // copy filter to constant memory and image to cuda array for texture memory
    cudaMemcpyToSymbol(filter, h_filter, filter_size * filter_size * sizeof(float));
    cudaMemcpy2DToArray(d_img, 0, 0, img, cols * sizeof(float), cols * sizeof(float), rows, cudaMemcpyHostToDevice);

    // create resource description which holds array
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_img;

    // create texture description which defines access behavior
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    //kernel call
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid((rows + BLOCK_SIZE - 1) / BLOCK_SIZE, (cols + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    int tile_size = BLOCK_SIZE + (filter_size / 2) * 2;
    conv_2d_tex<<<dimGrid, dimBlock, tile_size * tile_size * sizeof(float), 0>>>(texObj, d_out, rows, cols, filter_size);
    // cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ = 0.f;
    cudaEventElapsedTime(&time_, start, stop);
    printf("Texture elapsed time (ms): %.3f \n", time_);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(out, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    // print_matrix(out, rows, cols);

    cudaDestroyTextureObject(texObj);

    // frees
    cudaFreeArray(d_img);
    cudaFree(d_out);
    cudaFreeHost(img);
    cudaFreeHost(h_filter);
    free(out);

    return 0;
}

