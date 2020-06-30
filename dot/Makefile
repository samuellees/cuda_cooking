SOURCE_CUDA = dot.cu
OBJECT_CUDA = dot.o
SOURCE_CPP = dot.cpp dot_main.cpp

dot_main : $(OBJECT_CUDA)
	g++ -o dot_main $(SOURCE_CPP) $(OBJECT_CUDA) -lcudart -L/usr/local/cuda/lib64 -std=c++11

$(OBJECT_CUDA) :
	nvcc -c $(SOURCE_CUDA)

.PHONY : clean
clean :
	-rm dot_main ./*.o
