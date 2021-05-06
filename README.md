

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

## 如果这个项目对你有所帮助，请赞助作者，创作更高质量的干货．其他合作，可QQ:1371117942联系．赞助扫码下图：
![paycode](https://user-images.githubusercontent.com/9131273/117252210-3ac17e80-ae78-11eb-8b8a-8a5fdd89a4bd.png)


