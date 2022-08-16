# CUDA_Conv2d
Optimized simple implementation of a convolution, with a few different attempted methods and comparison.

## Table of Contents
* [Introduction](#introduction)
* [Shared Memory Implementation](#shared-memory-implementation)
* [Warp Synchronous Shuffle Implementation](#warp-synchronous-shuffle-implementation)
* [Texture Memory Implementation](#texture-memory-implementation)

## Introduction
The purpose of this project is just to practice CUDA programming and generally optimized use of GPU architecture. The structure of the kernel is a straightforward elementwise convolution, as opposed to more advanced algorithms which take advantage of highly optimized SGEMM kernels. Each thread is responsible for one element in the output (so a grid completely covering the input image is used). Each of these threads iteratively retrieves the filter and associated input values, multiplies and accumulates them. 

Both the shared memory and warp synchronous shuffle implementations take advantage of shared memory for holding the input array, while each implementation uses constant memory to hold the filter (as each value will be read one at a time and broadcast to every thread). The texture memory implementation experiments with texture memory for holding the input array, though is not as effective. The warp shuffle approach attempts to optimize the repeated load of the same values from shared memory by using shuffling with modest success.

An additional method for loading the input array to shared memory (outlined in the warp shuffle section) was tested which reduced divergent branches but failed to improve performance.

## Shared Memory Implementation
Since the input is going to be read from repeatedly, instead of loading from global memory, the kernel will make use of shared memory to hold the array. 

### Loading into shared memory
If you picture the grid covering the input array, each thread block will load its corresponding section, plus the "padding" areas. These are at the edges of each block where either there is padding (outside the array) or elements from the input. Becuase the area that must be loaded is larger than the block size, each thread must load more than one element. This is done with a "block-strided" load, stepping first in columns, then rows, which results in 4 total iterations for small filter sizes. The accesses from global memory are fully coalesced in the first iteration, but may have uncoalesced access once strided. Additionally, there will be a large number of inactive warps (because they would access outside what the block needs to load).

```c++
for (int tile_row = 0; tile_row < tile_size / BLOCK_SIZE; ++tile_row) {
            int tile_y = threadIdx.y + tile_row * BLOCK_SIZE;
            int overall_y = tid_y - padding + tile_row * BLOCK_SIZE;
        for (int tile_col = 0; tile_col < tile_size / BLOCK_SIZE; ++tile_col) {
            // three regions - inside array, outside array inside padding, outside padding
            int tile_x = threadIdx.x + tile_col * BLOCK_SIZE;
            if (tile_x >= tile_size || tile_y >= tile_size) continue;  // outside padding/(array where thread should read from)

            // everything else will be inside array or padding
            int overall_x = tid_x - padding + tile_col * BLOCK_SIZE;
            s_input[tile_y * tile_size + tile_x] = (overall_x >= 0 && overall_x < cols && overall_y >= 0 && overall_y < rows) ? \
                input[overall_y * cols + overall_x] : 0.f;
        }
    }
    __syncthreads();
```
### Convolution
The multiplication is done by iterating throught the filter, loading the values to each thread, loading the corresponding input value from shared memory, multiplying and accumulating in a local variable, before writing to the output array. Each warp accesses consecutive elements from shared memory and thus for a block size of 32, is fully coalesced with no bank conflicts (for a smaller block size, the padding would result in a small displacement, leading to a few bank conflicts). It is, however, repeatedly loading the same values, as the x value of each load only increases by 1. The warp synchronous method attempts to exploit this.
```c++
bool inside_array = tid_x >= 0 && tid_x < cols && tid_y >= 0 && tid_y < rows;
for (int fid_y = 0; fid_y < filter_size; ++fid_y) {
    int idx_y = threadIdx.y + fid_y;
    for (int fid_x = 0; fid_x < filter_size; ++fid_x) {
        int idx_x = threadIdx.x + fid_x;
        val += inside_array ? s_input[idx_y * tile_size + idx_x] * filter[fid_y * filter_size + fid_x] : 0.f;
    }
}
```

## Warp Synchronous Shuffle Implementation
This method attempts to improve the repeated loads from shared memory, since each load loads most of the same elements, though in different threads. For a block size of 32, as the x index of the filter increases, the warp loads only 1 new value, the rest can be shuffled from the warp with relative warp id = +1. It is only written to handle a block size of 32 and a filter size of 3, though it could be abstracted to work on higher filter sizes easily.
### Convolution with shuffling
For this, each thread in the warp will load one element. With a filter size of 3, there will be 1 padding row and column on each side, or 2 elements of total padding. So, there are 2 more elements that need to be loaded, which will be done with a vectorized load by the last thread in the warp. Warp synchronous shuffles will then propogate values down to the lower threads. This reduces a lot of repeated loads from shared memory, but with this approach, it is hard to completely avoid it (as the last couple elements are still needed). Overall, for these parameters, the compute time is reduced by about 10%.
```c++
bool inside_array = tid_x >= 0 && tid_x < cols && tid_y >= 0 && tid_y < rows;
float2 next_vals;
for (int fid_y = 0; fid_y < filter_size; ++fid_y) {
    int idx_y = threadIdx.y + fid_y;
    float arr_val = s_input[idx_y * tile_size + threadIdx.x];
    if (threadIdx.x == 31) next_vals = reinterpret_cast<float2*>(s_input)[32];
    val += inside_array ? arr_val * filter[fid_y * filter_size] : 0.f;
    arr_val = __shfl_down_sync(FULL_MASK, arr_val, 1);
    next_vals.x = __shfl_down_sync(FULL_MASK, arr_val, 1);
    for (int fid_x = 1; fid_x < filter_size; ++fid_x) {
        val += inside_array ? arr_val * filter[fid_y * filter_size + fid_x] : 0.f;
        arr_val = next_vals.x;
    }
}
```

## Texture Memory Implementation
Texture memory is slightly nicer, in that the kernel doesn't have to manage the padding, but reduces performance compared to the shared memory approach (since the shared memory accesses are already fully coalesced). The kernel is the same, except for setting up texture memory and loading from texture instead of shared memory.
