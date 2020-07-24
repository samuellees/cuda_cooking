SOURCE_CUDA = conv.cu
OBJECT_CUDA = conv.o
SOURCE_CPP = conv_main.cpp

LINK_LIBS = -lcudart -lcublas -lcudnn
LINK_DIR = -L/usr/local/cuda/lib64
INCLUDE_DIR = -I/usr/local/cuda/include

main : $(OBJECT_CUDA)
	g++ -o conv_main $(SOURCE_CPP) $(OBJECT_CUDA) $(LINK_LIBS) $(LINK_DIR) $(INCLUDE_DIR) -std=c++11

$(OBJECT_CUDA) :
	nvcc -c $(SOURCE_CUDA) -std=c++11 -arch=sm_61
	# nvcc -c $(SOURCE_CUDA) -std=c++11 -arch=sm_61 --ptxas-options=-v

.PHONY : clean
clean :
	-rm conv_main ./*.o
