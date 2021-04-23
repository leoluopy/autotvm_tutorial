import sys, os
import time

import cv2
import torch
import tvm
import numpy as np
from tvm import relay, autotvm
from tvm.contrib import graph_executor
import tvm.contrib.graph_executor as runtime

from centerface import InitCenterFacePy, GetBoxLandMarks
from tune_relay_cuda import tune_tasks


def relay_import_from_torch(model, direct_to_mod_param=False):
    input_shape = [1, 3, 544, 960]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    if direct_to_mod_param:
        return mod, params

    # target = tvm.target.Target("llvm", host="llvm")
    # dev = tvm.cpu(0)
    target = tvm.target.cuda()
    dev = tvm.device(str(target), 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    tvm_model = graph_executor.GraphModule(lib["default"](dev))

    return tvm_model, dev


def case_default_relay_centerFace():
    model = InitCenterFacePy()
    frame = cv2.imread("../../ims/scene.jpg")
    # frame = cv2.imread("../../data_set/demo10set/13.jpg")
    dets, lms, poses = GetBoxLandMarks(frame)
    # visulise_center_pose(dets, lms, poses, frame)

    tvm_centerFaceModel, dev = relay_import_from_torch(model.module.cpu())

    input_tensor, img_h_new, img_w_new, scale_w, scale_h = centerFacePreprocess(frame)
    tvm_centerFaceModel.set_input("input0", tvm.nd.array(input_tensor.astype("float32")))
    tvm_centerFaceModel.run()
    heatmap, scale, offset, lms = torch.tensor(tvm_centerFaceModel.get_output(0).asnumpy()), \
                                  torch.tensor(tvm_centerFaceModel.get_output(1).asnumpy()), \
                                  torch.tensor(tvm_centerFaceModel.get_output(2).asnumpy()), \
                                  torch.tensor(tvm_centerFaceModel.get_output(3).asnumpy())

    dets, lms, poses = centerFacePostProcess(heatmap, scale, offset, lms, img_h_new, img_w_new, scale_w, scale_h)
    # visulise_center_pose(dets, lms, poses, frame)

    print("start profiling the time")
    tvm_centerFaceModel.set_input("input0", tvm.nd.array(input_tensor.astype("float32")))
    ftimer = tvm_centerFaceModel.module.time_evaluator("run", dev, number=1, repeat=600)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )

    model = model.cuda()
    input_tensor = torch.tensor(input_tensor).cuda()
    t0 = time.time()
    for i in range(600):
        out = model(input_tensor)
    print("torch Mean time cost:{}".format((time.time() - t0) / 600.))


def profile_autvm_centerFace(mod, target, params, input_shape, dtype, log_file):
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("input0", data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=100)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
        lib.export_library("centerFace_relay.so")


def case_eval_from_autotvmlog():
    model = InitCenterFacePy()

    log_file = "centerFace.log"  #
    dtype = "float32"
    print("Extract tasks centerFace...")
    mod, params, = relay_import_from_torch(model.module.cpu(), direct_to_mod_param=True)
    input_shape = [1, 3, 544, 960]
    target = tvm.target.cuda()
    profile_autvm_centerFace(mod, target, params, input_shape, dtype, log_file)


def case_autotvm_relay_centerFace():
    model = InitCenterFacePy()

    log_file = "centerFace.log"
    dtype = "float32"
    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        # "n_trial": 1,
        "n_trial": 2000,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    print("Extract tasks centerFace...")
    mod, params, = relay_import_from_torch(model.module.cpu(), direct_to_mod_param=True)
    input_shape = [1, 3, 544, 960]
    target = tvm.target.cuda()
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    profile_autvm_centerFace(mod, target, params, input_shape, dtype, log_file)


if __name__ == '__main__':
    # case_default_relay_centerFace()
    case_autotvm_relay_centerFace()
    # case_eval_from_autotvmlog()
    print("END")
