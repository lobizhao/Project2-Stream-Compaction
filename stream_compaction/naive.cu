#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void naiveScan(int n, int d, int *odata, const int *idata) {

            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            //round depth 
            int dOffset = 1 << (d - 1);

            if (index >= dOffset) {
                odata[index] = idata[index] + idata[index - dOffset];
            } else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            dim3 blockSize(128);
            dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
            
            int *input = dev_idata;
            int *output = dev_odata;
            
            for (int d = 1; d <= ilog2ceil(n); d++) {
                naiveScan<<<gridSize, blockSize>>>(n, d, output, input);
                std::swap(input, output);
            }
            
            cudaMemcpy(odata, input, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            
            timer().endGpuTimer();
        }
    }
}
