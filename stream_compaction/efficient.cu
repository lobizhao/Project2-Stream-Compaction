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


        // GPU only scan function for device arrays

        void scanOnGpu(int n, int *dev_odata, int *dev_idata) {
            int depth2n = 1 << ilog2ceil(n);
            int *dev_temp;
            cudaMalloc((void**)&dev_temp, depth2n * sizeof(int));
            cudaMemset(dev_temp, 0, depth2n * sizeof(int));
            cudaMemcpy(dev_temp, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            
            dim3 blockSize(128);
            dim3 gridSize((depth2n + blockSize.x - 1) / blockSize.x);
            
            //run Up sweep
            for (int d = 0; d < ilog2ceil(depth2n); d++) {
                upSweep<<<gridSize, blockSize>>>(depth2n, d, dev_temp);
            }
            
            cudaMemset(&dev_temp[depth2n - 1], 0, sizeof(int));
            
            //run Down sweep
            for (int d = ilog2ceil(depth2n) - 1; d >= 0; d--) {
                downSweep<<<gridSize, blockSize>>>(depth2n, d, dev_temp);
            }
            
            cudaMemcpy(dev_odata, dev_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(dev_temp);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int depth2n = 1 << ilog2ceil(n);
            int *dev_data;
            cudaMalloc((void**)&dev_data, depth2n * sizeof(int));
            cudaMemset(dev_data, 0, depth2n * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            dim3 blockSize(128);
            dim3 gridSize((depth2n + blockSize.x - 1) / blockSize.x);
            
            //run Up sweep
            for (int d = 0; d < ilog2ceil(depth2n); d++) {
                upSweep<<<gridSize, blockSize>>>(depth2n, d, dev_data);
            }
            
            cudaMemset(&dev_data[depth2n - 1], 0, sizeof(int));
            
            //run Down sweep
            for (int d = ilog2ceil(depth2n) - 1; d >= 0; d--) {
                downSweep<<<gridSize, blockSize>>>(depth2n, d, dev_data);
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
            
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            dim3 blockSize(128);
            dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

            //map->scan->scatter!!!!          
            StreamCompaction::Common::kernMapToBoolean<<<gridSize, blockSize>>>(n, dev_bools, dev_idata);
            
            scanOnGpu(n, dev_indices, dev_bools);

            StreamCompaction::Common::kernScatter<<<gridSize, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            
            int finalBool, finalIndex;
            cudaMemcpy(&finalBool, &dev_bools[n-1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&finalIndex, &dev_indices[n-1], sizeof(int), cudaMemcpyDeviceToHost);
            int count = finalIndex + finalBool;
            
            if (count > 0) {
                cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            
            timer().endGpuTimer();
            return count;
        }
    }
}
