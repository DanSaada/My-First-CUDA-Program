# My-First-CUDA-Program
A simple CUDA program that demonstrates its basic principles.

<img src="https://github.com/DanSaada/My-First-CUDA-Program/assets/112869076/d69017a2-ce7a-4ed5-9f1e-008f62f6fa55" width="500">

## Introduction
CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. 
It allows software developers and software engineers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing – an approach known as GPGPU (General-Purpose computing on Graphics Processing Units).
CUDA is extensively used in fields requiring intensive mathematical computations like deep learning, scientific computing, and engineering simulations. 
By using CUDA, applications can utilize the massively parallel processing power of NVIDIA GPUs to achieve greater performance compared to CPU-only computing.


## Implementation
* Example 1: First, we have to turn our add function into a function that the GPU can run, called a kernel in CUDA.
To do this, all we have to do is add the specifier __global__ to the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.
To compute on the GPU, we need to allocate memory accessible by the GPU. Unified Memory in CUDA makes this easy by providing a single memory space accessible by all GPUs and CPUs in your system.
To allocate data in unified memory, call cudaMallocManaged(), which returns a pointer that you can access from host (CPU) code or device (GPU) code. To free the data, just pass the pointer to cudaFree().
Finally, we need to launch the add() kernel, which invokes it on the GPU. CUDA kernel launches are specified using the triple angle bracket syntax <<< >>>. We just have to add it to the call to add before the parameter list.
add<<<1, 1>>>: This specifies the execution configuration of the CUDA kernel. 1, 1 here means that the kernel is launched with 1 block and each block contains 1 thread.

![image](https://github.com/DanSaada/My-First-CUDA-Program/assets/112869076/a07ab74c-ffb3-4ed3-83c9-1786d1517c14)

* Example 2: Now that we have run a kernel with one thread that does some computation, how do we make it parallel? 
The key is in CUDA’s <<<1, 1>>>syntax. This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU.
If we run the code with only this change, it will do the computation once per thread, rather than spreading the computation across the parallel threads. To do it properly, we need to modify the kernel.
CUDA C++ provides keywords that let kernels get the indices of the running threads (threadIdx.x and blockDim.x).

![image](https://github.com/DanSaada/My-First-CUDA-Program/assets/112869076/3c3ba0af-f8bb-48b7-8e0e-3019951ff701)


* Example 3: CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. Each SM can run multiple concurrent thread blocks.
To take full advantage of all these threads, we should launch the kernel with multiple thread blocks.
Together, the blocks of parallel threads make up what is known as the grid. Since we have N elements to process, and 256 threads per block, we just need to calculate the number of blocks to get at least N threads.
We simply divide N by the block size (being careful to round up in case N is not a multiple of blockSize).
We also need to update the kernel code to take into account the entire grid of thread blocks.
The idea is that each thread gets its index by computing the offset to the beginning of its block (the block index times the block size: blockIdx.x * blockDim.x) and adding the thread’s index within the block (threadIdx.x).
The code blockIdx.x * blockDim.x + threadIdx.x is idiomatic CUDA.

![image](https://github.com/DanSaada/My-First-CUDA-Program/assets/112869076/b1fff216-3aaf-4c77-b687-e8d343c23479)

![image](https://github.com/DanSaada/My-First-CUDA-Program/assets/112869076/01c2933d-aef6-41e0-ab46-cd0c3ba661ef)






## Installing And Executing
    
To clone and run this application, you'll need [Git](https://git-scm.com) installed on your computer.
  
From your command line:

  
```bash
# Clone this repository.
$ git clone https://github.com/DanSaada/My-First-CUDA-Program.git

# Go into the repository.
$ cd My-First-CUDA-Program

# Compile using the nvcc (CUDA C++ compiler)
$ nvcc example3.cu -o add_cuda

# Run the program.
 ./add_cuda
```

## Author
- [Dan Saada](https://github.com/DanSaada)

