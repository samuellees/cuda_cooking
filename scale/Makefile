SOURCE_CUDA = scale.cu
OBJECT_CUDA = scale.o

scale_main : $(OBJECT_CUDA)
	g++ -o scale_main $(OBJECT_CUDA) -lcudart -lcublas -L/usr/local/cuda/lib64 -std=c++11

$(OBJECT_CUDA) :
	nvcc -c $(SOURCE_CUDA) -std=c++11

.PHONY : clean
clean :
	-rm scale_main ./*.o
