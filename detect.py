from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "center"))
import torch
import tqdm
from center.models.model import load_model, create_model
from center.detector.detector import BaseDetector
from center.models.decode import point_decode
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from center.utils.config import Cfg
from center.utils.debugger import Debugger
import argparse


class CENTER_MODEL(object):
    def __init__(self, config_path):
        config = Cfg.load_config_from_file(config_path)
        self.weight_path = config['predictor']['weight']
        self.scale = config['predictor']['scale']
        self.threshold = config['predictor']['threshold']
        self.max_obj_predict = config['dataset']['max_object']
        self.num_classes = config['dataset']['num_classes']
        self.arch = config['model']['arch']
        self.heads = config['model']['heads']
        self.head_conv = config['model']['head_conv']

        self.model = create_model(arch=self.arch, heads=self.heads, head_conv=self.head_conv)
        self.model = load_model(self.model, self.weight_path)


        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.dt = BaseDetector(config)
        self.debugger = Debugger(num_classes=self.num_classes)


    def quantization_model(self):
        model_quanti = torch.quantization.quantize_dynamic(self.model, dtype=torch.qint8)
        sm = torch.jit.script(model_quanti)
        torch.jit.save(sm, "vgg_seq2seq_quanti.pt")
        self.model = model_quanti

    def detect_obj(self, img):
        """
        Fix resolution image
        :param img: cv2 image
        :return:
        """
        image, meta = self.dt.pre_process(img, self.scale)  # size image change to 512x512x3
        # from IPython import embed; embed();
        with torch.no_grad():
            if torch.cuda.is_available():
                image = image.cuda()
            else:
                image = image.to(torch.device('cpu'))
            # start = time.time()
            output = self.model(image)[-1]
            # print("Time predict: ", time.time() - start)
            hm = output['hm'].sigmoid_()
            reg = output['reg']
            dets = point_decode(hm, reg=reg, K=self.max_obj_predict)

        dets = self.dt.post_process(dets, meta)
        dets = [dets]
        results = self.dt.merge_outputs(dets)

        list_center = []
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[2] >= self.threshold:
                    x_center, y_center = max(int(bbox[0]), 0), max(0, int(bbox[1]))
                    list_center.append([x_center, y_center])
        have_cmnd = True
        if len(list_center) >= 4:
            points = self.order_points(np.array(list_center[:4]))
        else:
            print("Cannot detect 4 corners !!!, Number of conners detected was ", len(list_center))
            have_cmnd = False
            return img, have_cmnd
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_aligh = self.align(img, points)
        return img_aligh, have_cmnd


    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def align(self, image, pts):
        pts = np.array(pts, dtype="float32")
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/cmnd.yml')
    parser.add_argument("--image_path", type=str, default='/home/hisiter/IT/5_year/Graduation_Thesis /Generic_OCR/ocr_deploy/Object_Corner_Detection/img_test/cmnd.jpg')
    args = parser.parse_args()
    config_path = args.config
    # config = Cfg.load_config_from_file()
    # print(config)
    model = CENTER_MODEL(config_path)
    model.quantization_model()
    # one image
    img = cv2.imread(args.image_path)
    t1 = time.time()
    cmnd_cropped, have_cmnd = model.detect_obj(img)
    print("Time centernet: ", time.time() - t1)

    plt.imshow(cmnd_cropped)
    plt.show()


