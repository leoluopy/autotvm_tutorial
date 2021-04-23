import sys, os
import time

import cv2
import torch
import tvm
import tvm.contrib.graph_executor as runtime
import numpy as np

from RealTimeEval.wiseSoft.externalDet.centerfacePose.centerface_pose import centerPosePreprocess, \
    centerPosePostProcess, visulise_center_pose

if __name__ == '__main__':
    frame = cv2.imread("../../ims/scene.jpg")

    target = tvm.target.cuda()
    dev = tvm.device(str(target), 0)

    lib = tvm.runtime.load_module("centerpose_relay.so")
    tvm_centerPoseModel = runtime.GraphModule(lib["default"](dev))

    input_tensor, img_h_new, img_w_new, scale_w, scale_h = centerPosePreprocess(frame)
    input_tensor = np.zeros([1, 3, 544, 960], dtype=np.float32)
    tvm_centerPoseModel.set_input("input0", tvm.nd.array(input_tensor.astype("float32")))

    # while True:
    t0 = time.time()
    tvm_centerPoseModel.run()
    print("tvm inference cost: {}".format(time.time() - t0))

    heatmap, scale, offset, lms, yaw, pitch, roll = torch.tensor(tvm_centerPoseModel.get_output(0).asnumpy()), \
                                                    torch.tensor(tvm_centerPoseModel.get_output(1).asnumpy()), \
                                                    torch.tensor(tvm_centerPoseModel.get_output(2).asnumpy()), \
                                                    torch.tensor(tvm_centerPoseModel.get_output(3).asnumpy()), \
                                                    torch.tensor(tvm_centerPoseModel.get_output(4).asnumpy()), \
                                                    torch.tensor(tvm_centerPoseModel.get_output(5).asnumpy()), \
                                                    torch.tensor(tvm_centerPoseModel.get_output(6).asnumpy()),

    dets, lms, poses = centerPosePostProcess(heatmap, scale, offset, lms, yaw, pitch, roll,
                                             img_h_new, img_w_new, scale_w, scale_h, decodePose=True)
    visulise_center_pose(dets, lms, poses, frame)
    print("END")
