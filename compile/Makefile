
SOURCE_CPP = main.cpp once.cpp twice.cpp

main : $(OBJECT_CUDA)
	g++ -o main $(SOURCE_CPP) -std=c++11

.PHONY : clean
clean :
	-rm main ./*.o
