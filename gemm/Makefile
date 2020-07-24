SOURCE_CUDA = gemm.cu
OBJECT_CUDA = gemm.o
SOURCE_CPP = gemm_main.cpp

main : $(OBJECT_CUDA)
	g++ -o gemm_main $(SOURCE_CPP) $(OBJECT_CUDA) -lcudart -lcublas -L/usr/local/cuda/lib64 -std=c++11

$(OBJECT_CUDA) :
	nvcc -c $(SOURCE_CUDA) -std=c++11 -arch=sm_61
	# nvcc -c $(SOURCE_CUDA) -std=c++11 -arch=sm_61 --ptxas-options=-v

.PHONY : clean
clean :
	-rm gemm_main ./*.o
