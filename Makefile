SOURCE_CUDA = gemm.cu
OBJECT_CUDA = gemm.o
SOURCE_CPP = main.cpp

main : $(OBJECT_CUDA)
	g++ -o main $(SOURCE_CPP) $(OBJECT_CUDA) -lcudart -L/usr/local/cuda/lib64

$(OBJECT_CUDA) :
	nvcc -c $(SOURCE_CUDA)

.PHONY : clean
clean :
	-rm main ./*.o
