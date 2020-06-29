# cuda_matrix_multiply
Learning CUDA: Implement and optimize gemv


1. 全局内存版本
使用64个线程块*256个线程进行网格跨步，每个线程独立完成若干个结果的计算，线程之间无通信。总时间3.5s。


