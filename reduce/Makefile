SOURCE_CUDA = reduce.cu
OBJECT_CUDA = reduce.o
SOURCE_CPP = reduce_main.cpp

LINK_LIBS = -lcudart
LINK_DIR = -L/usr/local/cuda/lib64
INCLUDE_DIR = -I/usr/local/cuda/include
# LINK_DIR = -L/gpfs/share/software/cuda/10.1/lib64
# INCLUDE_DIR = -I/gpfs/share/software/cuda/10.1/include

main : $(OBJECT_CUDA)
	g++ -o reduce_main $(SOURCE_CPP) $(OBJECT_CUDA) $(LINK_LIBS) $(LINK_DIR) $(INCLUDE_DIR) -std=c++11

$(OBJECT_CUDA) :
	nvcc -c $(SOURCE_CUDA) -std=c++11 -arch=sm_61
	# nvcc -c $(SOURCE_CUDA) -std=c++11 -arch=sm_61 -Xptxas -v -maxrregcount=10

.PHONY : clean
clean :
	-rm reduce_main ./*.o
