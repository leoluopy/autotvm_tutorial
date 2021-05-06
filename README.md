![paycode](https://user-images.githubusercontent.com/9131273/117250302-976f6a00-ae75-11eb-8b33-a64bf755307b.png)
![paycode](https://user-images.githubusercontent.com/9131273/117250318-9fc7a500-ae75-11eb-849b-04ebb75b24a8.png)
# autotvm_tutorial
autoTVM神经网络推理代码优化搜索演示，基于tvm编译开源模型centerface，并使用autoTVM搜索最优推理代码，　最终部署编译为c++代码，演示平台是cuda，可以是其他平台，例如树莓派，安卓手机，苹果手机

+ 知乎介绍文章位置：　https://zhuanlan.zhihu.com/p/366913595

> Thi is a demonstration of how to use autoTVM to search and optimize a neural network inference code. the main process of this program is , firstly use tvm to compile opensource model centerface , then use autotvm to auto search the best inference code for the compiled model, finaly the model to compile and deploy to c++ inference code , the demonstration platform is cuda framework , alternatively other platform is acceptable , ie rasspery , android and apple



# NOTE:
+ add variables "PATH=$PATH:/usr/local/cuda-11.1/bin" in order to use nvcc

# HOW TO

1. python tuning_centerface.py
2. use function "case_eval_from_autotvmlog()" in tuning_centerface.py to generate the inference dynamic library which is searched by the autoTVM
3. python inference_relay.py to verify the result
4. convert to the c++ api , see cpp_deploy project
