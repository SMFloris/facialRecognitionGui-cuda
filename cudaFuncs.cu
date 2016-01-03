#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define NUM_BANKS 8
#define LOG_NUM_BANKS 3
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void proprietati()
{
        int devCount;
        cudaGetDeviceCount(&devCount);
        printf("Avem %d device-uri.\n", devCount);

        for (int i = 0; i < devCount; ++i)
        {
            // Get device properties
            printf("\nCUDA Device #%d\n", i);
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, i);
            printDevProp(devProp);
        }
}

__global__ void vecAdd(float *a, float *b, float *c, unsigned int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
__global__ void vecSub(float *a, float *b, float *c, unsigned int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

__global__ void eulVector(float *a, float *b,float *c, unsigned int n)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < n){
        c[id] = a[id] - b[id];
        c[id]*= c[id];
    }
}

__global__ void smallSum(unsigned char *out, unsigned char *in, int n)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(threadIdx.x == 1023) {
        out[id/1024] = id;
    }
}

// reduction sum
__global__ void reductionSum(float *in_array, float *out_array, unsigned long size)
{
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    shared[tid] = in_array[i];
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if(tid==0) //first thread, last sum
    {
        out_array[blockIdx.x]=shared[0];
    }
}

__global__ void cumSum(float *in_array, float *out_array, unsigned long size)
{
    extern __shared__ float shared[];

    unsigned int tid = threadIdx.x;
    unsigned int b_offset = blockIdx.x * blockDim.x;
    unsigned int offset = 1;

    int i = tid;
    int j = tid + blockDim.x / 2;
    int offset_i = CONFLICT_FREE_OFFSET(i);
    int offset_j = CONFLICT_FREE_OFFSET(j);
    shared[i + offset_i] = in_array[i + b_offset];
    shared[j + offset_j] = in_array[j + b_offset];

    // scan up
    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        __syncthreads();

        if (tid < s) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            i += CONFLICT_FREE_OFFSET(i);
            j += CONFLICT_FREE_OFFSET(j);
            shared[j] += shared[i];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        shared[blockDim.x - 1 + CONFLICT_FREE_OFFSET(blockDim.x - 1)] =
            0;
    }
    // scan down
    for (int s = 1; s < blockDim.x; s <<= 1) {
        offset >>= 1;
        __syncthreads();

        if (tid < s) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            i += CONFLICT_FREE_OFFSET(i);
            j += CONFLICT_FREE_OFFSET(j);
            float tmp = shared[i];
            shared[i] = shared[j];
            shared[j] += tmp;
        }
    }
    __syncthreads();
    out_array[i] = shared[i + offset_i];
    out_array[j] = shared[j + offset_j];
}

void test1(float *a, float *b, unsigned long n)
{
    float *d_a, *d_b, *d_rez;
    float *rez = (float*) malloc(n*sizeof(float));

    cudaMalloc((void **)&d_a,n*sizeof(float));
    cudaMalloc((void **)&d_b,n*sizeof(float));
    cudaMalloc((void **)&d_rez,n*sizeof(float));

    cudaMemcpy(d_a,a,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,n*sizeof(float),cudaMemcpyHostToDevice);

    vecAdd<<<161,64>>>(d_a,d_b,d_rez,n);

    cudaMemcpy(rez,d_rez,10304*sizeof(float),cudaMemcpyDeviceToHost);

    printf("\n REZULTAT TEST: %f \n",rez[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_rez);

    free(rez);
}

// a imagini
// b imagine
float euclideanNormAsync(float *a, float *b, unsigned long n, unsigned long nrPoze)
{
    float *d_a,*d_b,*d_rez,*d_temp;
    float *sum;

    cudaStream_t stream;

    cudaStreamCreate(&stream);

    cudaHostRegister((void **)&a, n*nrPoze*sizeof(float), cudaHostRegisterPortable);
    cudaHostAlloc((void **)&sum, nrPoze*sizeof(float), cudaHostAllocDefault);

    cudaMalloc((void **)&d_a,n*sizeof(float));
    cudaMalloc((void **)&d_b,n*sizeof(float));
    cudaMalloc((void **)&d_temp,n*sizeof(float));
    cudaMalloc((void **)&d_rez,(n/64+(256-n/64))*sizeof(float));

    //cudaMemcpy(d_a,a,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,n*sizeof(float),cudaMemcpyHostToDevice);
    for(unsigned long int i=0, k=0; i<n, k< nrPoze; i+= n, k++)
    {
        cudaMemcpyAsync(d_a,a+i,n*sizeof(float),cudaMemcpyHostToDevice, stream);

        eulVector<<<n/64, 64, n*sizeof(float)>>>(d_a,d_b,d_temp, n);
        reductionSum<<<n/64, 64, n*sizeof(float), stream>>>(d_temp,d_rez, n);
        cudaMemsetAsync(d_rez+n/64,0,95*sizeof(float),stream);
        reductionSum<<<1,256, n*sizeof(float), stream>>>(d_rez,d_temp,n);
        //cumSum<<<n/64, n/64, n*sizeof(float)>>>(d_rez,d_temp, n/64);

        cudaMemcpyAsync(sum+k,d_temp,sizeof(float),cudaMemcpyDeviceToHost, stream);
    }



    cudaStreamSynchronize(stream);

    float min = sum[0];
    for(int i=0;i<nrPoze;i++)
    {
        if(sum[i]<min)
            min=sum[i];
    }
    printf("%f ",sqrtf(min));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_temp);
    cudaFree(d_rez);

    cudaFreeHost(sum);
    cudaHostUnregister(a);

    cudaStreamDestroy(stream);



    return sqrtf(min);
}
