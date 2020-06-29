SOURCE_CUDA = gemv.cu
OBJECT_CUDA = gemv.o
SOURCE_CPP = gemv.cpp gemv_main.cpp

gemv_main : $(OBJECT_CUDA)
	# g++ -o gemv_main $(SOURCE_CPP) $(OBJECT_CUDA) -lcudart -L/usr/local/cuda/lib64 -std=c++11
	g++ -o gemv_main $(SOURCE_CPP) $(OBJECT_CUDA) -lcudart -L/gpfs/share/software/cuda/cuda-9.0/lib64 -std=c++11

$(OBJECT_CUDA) :
	nvcc -c $(SOURCE_CUDA)

.PHONY : clean
clean :
	-rm gemv_main ./*.o
