#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //up sweep
        __global__ void upSweep(int n, int d, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k < n && k % (1 << (d + 1)) == 0) {
                data[k + (1<<(d+1)) - 1] += data[k + (1<<d) - 1];
            }
        }

        //down sweep
        __global__ void downSweep(int n, int d, int *data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k < n && k % (1 << (d + 1)) == 0) {
                int temp = data[k + (1 << d) - 1];
                data[k + (1 << d) - 1] = data[k + (1 << (d + 1)) - 1];
                data[k + (1 << (d + 1)) - 1] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int log2n = 1 << ilog2ceil(n);
            int *dev_data;
            cudaMalloc((void**)&dev_data, log2n * sizeof(int));
            cudaMemset(dev_data, 0, log2n * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            dim3 blockSize(128);
            dim3 gridSize((log2n + blockSize.x - 1) / blockSize.x);
            
            //run Up sweep
            for (int d = 0; d < ilog2ceil(log2n); d++) {
                upSweep<<<gridSize, blockSize>>>(pow2n, d, dev_data);
            }
            
            cudaMemset(&dev_data[log2n - 1], 0, sizeof(int));
            
            //run Down sweep
            for (int d = ilog2ceil(log2n) - 1; d >= 0; d--) {
                downSweep<<<gridSize, blockSize>>>(pow2n, d, dev_data);
            }
            
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
            
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
