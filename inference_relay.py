import sys, os
import time

import cv2
import torch
import tvm
import tvm.contrib.graph_executor as runtime
import numpy as np

from centerface import centerFacePreprocess, centerFacePostProcess, centerFaceWriteOut

if __name__ == '__main__':
    frame = cv2.imread("./ims/6.jpg")

    target = tvm.target.cuda()
    dev = tvm.device(str(target), 0)

    lib = tvm.runtime.load_module("centerFace_relay.so")
    tvm_centerPoseModel = runtime.GraphModule(lib["default"](dev))

    input_tensor, img_h_new, img_w_new, scale_w, scale_h, raw_scale = centerFacePreprocess(frame)
    tvm_centerPoseModel.set_input("input0", tvm.nd.array(input_tensor.astype("float32")))

    for i in range(100):
        # 推理速率演示，推理多次后时间会稳定下来
        t0 = time.time()
        tvm_centerPoseModel.run()
        print("tvm inference cost: {}".format(time.time() - t0))

    heatmap, scale, offset, lms = torch.tensor(tvm_centerPoseModel.get_output(0).asnumpy()), \
                                  torch.tensor(tvm_centerPoseModel.get_output(1).asnumpy()), \
                                  torch.tensor(tvm_centerPoseModel.get_output(2).asnumpy()), \
                                  torch.tensor(tvm_centerPoseModel.get_output(3).asnumpy()),

    dets, lms = centerFacePostProcess(heatmap, scale, offset, lms, img_h_new, img_w_new, scale_w, scale_h, raw_scale)

    centerFaceWriteOut(dets, lms, frame)
    print("END")
