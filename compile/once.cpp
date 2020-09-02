#include "header.h"

template<typename T> 
void myPrint(T& a) {
  std::cout << a <<std::endl;
}

template<>
void myPrint<int>(int& a) {
  std::cout << a <<std::endl;
}
