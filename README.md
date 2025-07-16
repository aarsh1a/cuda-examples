# CUDA Examples

This project contains several CUDA examples demonstrating basic operations using CUDA programming. The examples include vector addition, matrix multiplication, and basic image processing tasks.

## Project Structure

```
cuda-examples
├── src
│   ├── vector_addition.cu
│   ├── matrix_multiplication.cu
│   └── image_processing.cu
├── include
│   └── utils.cuh
├── CMakeLists.txt
├── .gitignore
└── README.md
```

## Examples

### Vector Addition

The `vector_addition.cu` file contains a CUDA kernel for adding two vectors. It includes functions for memory allocation on the GPU, data transfer from the host to the device, kernel execution, and copying results back to the host.

### Matrix Multiplication

The `matrix_multiplication.cu` file implements a CUDA kernel for multiplying two matrices. It includes functions for memory allocation, data transfer, kernel execution, and result retrieval.

### Image Processing

The `image_processing.cu` file provides a CUDA implementation for basic image processing tasks, such as filtering or edge detection. It includes functions for loading images, applying filters using CUDA kernels, and saving the processed images.

## Utilities

The `utils.cuh` header file contains utility functions and definitions used across the CUDA source files, including error checking and memory management functions.

## Build Instructions

To build the project, you will need CMake and a CUDA-capable GPU. Follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cuda-examples
   ```

2. Create a build directory:
   ```
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   make
   ```

## Running the Examples

After building the project, you can run the examples by executing the compiled binaries located in the `build` directory.

## Dependencies

- CUDA Toolkit (version X.X or higher)
- CMake (version X.X or higher)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.