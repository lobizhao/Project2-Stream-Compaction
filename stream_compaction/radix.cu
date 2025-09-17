#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // only get the 0th binary 
        __global__ void get0thBit(int n, int *bits, const int *data, int bitPos) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            bits[index] = (data[index] >> bitPos) & 1;
        }

        //get inverse array set 0th bit as 1
        __global__ void invert0thBits(int n, int *notBits, const int *bits) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            notBits[index] = 1 - bits[index];
        }


        __global__ void radixScatter(int n, int *odata, const int *idata, 
                                       const int *bits, const int *falseIndices, 
                                       const int *trueIndices, int numFalse) {

            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            
            if (bits[index] == 0) {
                odata[falseIndices[index]] = idata[index];
            } else {
                odata[numFalse + trueIndices[index]] = idata[index];
            }
        }


        void sort(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int* dev_input;
            int* dev_output;
            int* dev_bits;
            int* dev_notBits;
            int* dev_falseIndices;
            int* dev_trueIndices;
            
            cudaMalloc((void**)&dev_input, n * sizeof(int));
            cudaMalloc((void**)&dev_output, n * sizeof(int));
            cudaMalloc((void**)&dev_bits, n * sizeof(int));
            cudaMalloc((void**)&dev_notBits, n * sizeof(int));
            cudaMalloc((void**)&dev_falseIndices, n * sizeof(int));
            cudaMalloc((void**)&dev_trueIndices, n * sizeof(int));
            

            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            dim3 blockSize(BLOCK_SIZE);
            dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
            

            for (int bit = 0; bit < 32; bit++) {
                //get the 0th binary bit
                get0thBit <<<gridSize, blockSize>>>(n, dev_bits, dev_input, bit);
                
                //invert array b 
                invert0thBits <<<gridSize, blockSize>>>(n, dev_notBits, dev_bits);
                
                StreamCompaction::Efficient::scanOnGpu(n, dev_falseIndices, dev_notBits);             
                StreamCompaction::Efficient::scanOnGpu(n, dev_trueIndices, dev_bits);
                
                int lastNotBit;
                int lastFalseIndex;

                cudaMemcpy(&lastNotBit, &dev_notBits[n-1], sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastFalseIndex, &dev_falseIndices[n-1], sizeof(int), cudaMemcpyDeviceToHost);

                //sum offset 
                int numFalse = lastFalseIndex + lastNotBit;
                radixScatter <<<gridSize, blockSize>>>(n, dev_output, dev_input,
                                                        dev_bits, dev_falseIndices, 
                                                        dev_trueIndices, numFalse);
                
                int *temp = dev_input;
                dev_input = dev_output;
                dev_output = temp;
            }
            
            cudaMemcpy(odata, dev_input, n * sizeof(int), cudaMemcpyDeviceToHost);
            //dele

            cudaFree(dev_input);
            cudaFree(dev_output);
            cudaFree(dev_bits);
            cudaFree(dev_notBits);
            cudaFree(dev_falseIndices);
            cudaFree(dev_trueIndices);
            
            timer().endGpuTimer();
        }
    }
}