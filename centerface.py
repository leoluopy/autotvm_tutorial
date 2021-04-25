import sys, os
import time

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding, groups, bias, withRelu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(outc)
        self.withRelu = withRelu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.withRelu:
            x = F.relu(x)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding, groups, bias, withRelu=True):
        super(DeConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(inc, outc, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(outc)
        self.withRelu = withRelu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.withRelu:
            x = F.relu(x)
        return x


class CenterFace(nn.Module):
    def __init__(self):
        super(CenterFace, self).__init__()
        self.conv363 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1, groups=1, bias=False)
        self.conv366 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.conv369 = ConvBlock(32, 16, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)
        self.conv371 = ConvBlock(16, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv374 = ConvBlock(96, 96, kernel_size=3, stride=2, padding=1, groups=96, bias=False)
        self.conv377 = ConvBlock(96, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 379 branch
        self.conv379 = ConvBlock(24, 144, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv382 = ConvBlock(144, 144, kernel_size=3, stride=1, padding=1, groups=144, bias=False)
        self.conv385 = ConvBlock(144, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 388 branch
        self.conv388 = ConvBlock(24, 144, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv391 = ConvBlock(144, 144, kernel_size=3, stride=2, padding=1, groups=144, bias=False)
        self.conv394 = ConvBlock(144, 32, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 396 branch
        self.conv396 = ConvBlock(32, 192, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv399 = ConvBlock(192, 192, kernel_size=3, stride=1, padding=1, groups=192, bias=False)
        self.conv402 = ConvBlock(192, 32, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 405 branch
        self.conv405 = ConvBlock(32, 192, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv408 = ConvBlock(192, 192, kernel_size=3, stride=1, padding=1, groups=192, bias=False)
        self.conv411 = ConvBlock(192, 32, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 414 branch
        self.conv414 = ConvBlock(32, 192, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv417 = ConvBlock(192, 192, kernel_size=3, stride=2, padding=1, groups=192, bias=False)
        self.conv420 = ConvBlock(192, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 422 branch
        self.conv422 = ConvBlock(64, 384, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv425 = ConvBlock(384, 384, kernel_size=3, stride=1, padding=1, groups=384, bias=False)
        self.conv428 = ConvBlock(384, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 431 branch
        self.conv431 = ConvBlock(64, 384, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv434 = ConvBlock(384, 384, kernel_size=3, stride=1, padding=1, groups=384, bias=False)
        self.conv437 = ConvBlock(384, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 440 branch
        self.conv440 = ConvBlock(64, 384, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv443 = ConvBlock(384, 384, kernel_size=3, stride=1, padding=1, groups=384, bias=False)
        self.conv446 = ConvBlock(384, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 449 branch
        self.conv449 = ConvBlock(64, 384, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv452 = ConvBlock(384, 384, kernel_size=3, stride=1, padding=1, groups=384, bias=False)
        self.conv455 = ConvBlock(384, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 457 branch
        self.conv457 = ConvBlock(96, 576, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv460 = ConvBlock(576, 576, kernel_size=3, stride=1, padding=1, groups=576, bias=False)
        self.conv463 = ConvBlock(576, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 466 branch
        self.conv466 = ConvBlock(96, 576, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv469 = ConvBlock(576, 576, kernel_size=3, stride=1, padding=1, groups=576, bias=False)
        self.conv472 = ConvBlock(576, 96, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 475 branch
        self.conv475 = ConvBlock(96, 576, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv478 = ConvBlock(576, 576, kernel_size=3, stride=2, padding=1, groups=576, bias=False)
        self.conv481 = ConvBlock(576, 160, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 483 branch
        self.conv483 = ConvBlock(160, 960, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv486 = ConvBlock(960, 960, kernel_size=3, stride=1, padding=1, groups=960, bias=False)
        self.conv489 = ConvBlock(960, 160, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 492 branch
        self.conv492 = ConvBlock(160, 960, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv495 = ConvBlock(960, 960, kernel_size=3, stride=1, padding=1, groups=960, bias=False)
        self.conv498 = ConvBlock(960, 160, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)

        # 501 branch
        self.conv501 = ConvBlock(160, 960, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv504 = ConvBlock(960, 960, kernel_size=3, stride=1, padding=1, groups=960, bias=False)
        self.conv507 = ConvBlock(960, 320, kernel_size=1, stride=1, padding=0, groups=1, bias=False, withRelu=False)
        self.conv509 = ConvBlock(320, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

        # last scales
        self.conv515 = ConvBlock(96, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv522 = ConvBlock(32, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv529 = ConvBlock(24, 24, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv533 = ConvBlock(24, 24, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

        # last deconv
        self.conv512 = DeConvBlock(24, 24, kernel_size=2, stride=2, padding=0, groups=1, bias=False)
        self.conv519 = DeConvBlock(24, 24, kernel_size=2, stride=2, padding=0, groups=1, bias=False)
        self.conv526 = DeConvBlock(24, 24, kernel_size=2, stride=2, padding=0, groups=1, bias=False)

        self.conv536 = nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.conv538 = nn.Conv2d(24, 2, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.conv539 = nn.Conv2d(24, 2, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.conv540 = nn.Conv2d(24, 10, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        pass

    def forward(self, x):
        x = self.conv363(x)
        x = self.conv366(x)
        x = self.conv369(x)
        x = self.conv371(x)
        x = self.conv374(x)
        # b 代表头或者尾，根据网络图决定。
        b378 = self.conv377(x)

        b379 = self.conv379(b378)
        b379 = self.conv382(b379)
        b386 = self.conv385(b379)

        b387 = b378 + b386

        b388 = self.conv388(b387)
        b388 = self.conv391(b388)
        b388 = self.conv394(b388)

        b396 = self.conv396(b388)
        b396 = self.conv399(b396)
        b396 = self.conv402(b396)

        b404 = b388 + b396
        b405 = self.conv405(b404)
        b405 = self.conv408(b405)
        b413 = self.conv411(b405)
        b413 = b413 + b404

        b421 = self.conv414(b413)
        b421 = self.conv417(b421)
        b421 = self.conv420(b421)

        b422 = self.conv422(b421)
        b422 = self.conv425(b422)
        b429 = self.conv428(b422)
        b430 = b429 + b421

        b431 = self.conv431(b430)
        b431 = self.conv434(b431)
        b438 = self.conv437(b431)
        b439 = b438 + b430

        b440 = self.conv440(b439)
        b440 = self.conv443(b440)
        b447 = self.conv446(b440)
        b448 = b447 + b439

        b449 = self.conv449(b448)
        b449 = self.conv452(b449)
        b456 = self.conv455(b449)

        b457 = self.conv457(b456)
        b457 = self.conv460(b457)
        b464 = self.conv463(b457)
        b465 = b464 + b456

        b466 = self.conv466(b465)
        b466 = self.conv469(b466)
        b473 = self.conv472(b466)
        b474 = b473 + b465

        b475 = self.conv475(b474)
        b475 = self.conv478(b475)
        b482 = self.conv481(b475)

        b483 = self.conv483(b482)
        b483 = self.conv486(b483)
        b490 = self.conv489(b483)
        b491 = b490 + b482

        b492 = self.conv492(b491)
        b492 = self.conv495(b492)
        b499 = self.conv498(b492)
        b500 = b499 + b491

        b501 = self.conv501(b500)
        b501 = self.conv504(b501)
        b501 = self.conv507(b501)
        b501 = self.conv509(b501)
        b514 = self.conv512(b501)

        b517 = self.conv515(b474)
        b518 = b517 + b514

        b521 = self.conv519(b518)
        b524 = self.conv522(b413)
        b525 = b521 + b524

        b528 = self.conv526(b525)
        b531 = self.conv529(b387)
        b532 = b531 + b528

        b535 = self.conv533(b532)

        b536 = self.conv536(b535)
        b537 = F.sigmoid(b536)
        b538 = self.conv538(b535)
        b539 = self.conv539(b535)
        b540 = self.conv540(b535)

        return b537, b538, b539, b540


def transform(h, w):
    img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
    scale_h, scale_w = img_h_new / h, img_w_new / w
    return img_h_new, img_w_new, scale_h, scale_w


def postprocess(heatmap, lms, offset, scale, threshold, img_h_new, img_w_new, scale_w, scale_h):
    dets, lms = decode(heatmap, scale, offset, lms, (img_h_new, img_w_new), threshold=threshold)
    if len(dets) > 0:
        dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / scale_w, dets[:, 1:4:2] / scale_h
        lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / scale_w, lms[:, 1:10:2] / scale_h
    else:
        dets = np.empty(shape=[0, 5], dtype=np.float32)
        lms = np.empty(shape=[0, 10], dtype=np.float32)
    return dets, lms


def nms(boxes, scores, nms_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    suppressed = np.zeros((num_detections,), dtype=np.bool)

    keep = []
    for _i in range(num_detections):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep


def decode(heatmap, scale, offset, landmark, size, threshold=0.1):
    heatmap = np.squeeze(heatmap)
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > threshold)
    boxes, lms = [], []
    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
            lm = []
            for j in range(5):
                lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
            lms.append(lm)
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :]
        lms = np.asarray(lms, dtype=np.float32)
        lms = lms[keep, :]
    return boxes, lms


def case_time_test():
    model = CenterFace()

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    model.eval()
    im = torch.zeros((1, 3, 640, 480), dtype=torch.float32).cuda()
    for i in range(100):
        start = time.time()
        out = model(im)
        print("cost :{}".format(time.time() - start))


g_model = {}


def InitCenterFacePy(modelPath=os.path.dirname(os.path.abspath(__file__)) + "/centerface.pth"):
    global g_model
    g_model = CenterFace()
    g_model.load_state_dict(torch.load(modelPath))
    if torch.cuda.is_available():
        g_model = nn.DataParallel(g_model).cuda()
    g_model.eval()
    return g_model


def centerFacePreprocess(bgrImg):
    h = bgrImg.shape[0]
    w = bgrImg.shape[1]
    maxSide = max(h, w)
    if maxSide > 960:
        raw_scale = 960. / maxSide
        bgrImg = cv2.resize(bgrImg, (int(w * raw_scale), int(h * raw_scale)))
    else:
        raw_scale = 1.0
    h = bgrImg.shape[0]
    w = bgrImg.shape[1]
    img_h_new, img_w_new, scale_h, scale_w = transform(h, w)

    # input_tensor = np.zeros((1, 3, 800, 800), dtype=np.float32)
    input_tensor = cv2.dnn.blobFromImage(bgrImg, scalefactor=1.0, size=(img_w_new, img_h_new), mean=(0, 0, 0),
                                         swapRB=True, crop=False)
    return input_tensor, img_h_new, img_w_new, scale_w, scale_h, raw_scale


def centerFacePostProcess(heatmap, scale, offset, lms, img_h_new, img_w_new, scale_w, scale_h, raw_scale):
    dets, lms = postprocess(heatmap.detach().cpu().numpy(), lms.detach().cpu().numpy(), offset.detach().cpu().numpy(),
                            scale.detach().cpu().numpy(), 0.35, img_h_new, img_w_new, scale_w, scale_h)

    boundingBoxes = []
    landmarks = []
    for i, det in enumerate(dets):
        # offineh = 0#int((det[3] - det[2]) * 0.3)
        # offinew = int((det[1] - det[0]) * 0.3)
        boundingBoxes.append(np.array([det[0] / raw_scale
                                          , det[1] / raw_scale
                                          , det[2] / raw_scale
                                          , det[3] / raw_scale
                                          , det[4] / raw_scale
                                       ], dtype=np.float64))
        landmarks.append([
            [lms[i][0] / raw_scale, lms[i][1] / raw_scale],
            [lms[i][2] / raw_scale, lms[i][3] / raw_scale],
            [lms[i][4] / raw_scale, lms[i][5] / raw_scale],
            [lms[i][6] / raw_scale, lms[i][7] / raw_scale],
            [lms[i][8] / raw_scale, lms[i][9] / raw_scale],
        ])
    return boundingBoxes, torch.tensor(landmarks)


def GetBoxLandMarks(bgrImg):
    global g_model
    if g_model == {}:
        InitCenterFacePy()

    input_tensor, img_h_new, img_w_new, scale_w, scale_h, raw_scale = centerFacePreprocess(bgrImg)

    start = time.time()
    out = g_model(torch.tensor(input_tensor).cuda())
    print("    center face det cost :{}".format(time.time() - start))

    heatmap, scale, offset, lms = out
    return centerFacePostProcess(heatmap, scale, offset, lms, img_h_new, img_w_new, scale_w, scale_h, raw_scale)


def centerFaceWriteOut(dets, lms, frame):
    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)

    for lm in lms.detach().cpu().numpy():
        lm = lm.reshape([10, 1])
        for i in range(0, 5):
            cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)

    cv2.imwrite("{}.jpg".format(time.time()), frame)


if __name__ == '__main__':
    InitCenterFacePy()
    frame = cv2.imread("ims/6.jpg")
    dets, lms = GetBoxLandMarks(frame)

    centerFaceWriteOut(dets, lms, frame)

