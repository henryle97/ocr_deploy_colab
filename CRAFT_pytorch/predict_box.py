import math
import os
import time

import cv2
import torch
from skimage import io
from torch.autograd import Variable
from torch.backends import cudnn
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import craft_utils
from model.craft_utils import copyStateDict
from model.craft import CRAFT
import params as pr
from utils import imgproc, file_utils
from model.refinenet import RefineNet
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
class CraftDetection:
    def __init__(self):
        self.model = CRAFT()
        if pr.cuda:
            self.model.load_state_dict(copyStateDict(torch.load(pr.trained_model)))
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = False
        else:
            self.model.load_state_dict(copyStateDict(torch.load(pr.trained_model, map_location='cpu')))
        self.model.eval()

        self.refine_model = None
        if pr.refine:
            self.refine_model = RefineNet()
            if pr.cuda:
                self.refine_model.load_state_dict(copyStateDict(torch.load(pr.refiner_model)))
                self.refine_model = self.refine_net.cuda()
                self.refine_model = torch.nn.DataParallel(self.refine_model)
            else:
                self.refine_model.load_state_dict(copyStateDict(torch.load(pr.refiner_model, map_location='cpu')))

            self.refine_model.eval()
            pr.poly = True

    def text_detect(self, image, have_cmnd=True):
        time0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, pr.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=pr.mag_ratio)
        print(img_resized.shape)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if pr.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_model is not None:
            with torch.no_grad():
                y_refiner = self.refine_model(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()


        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, pr.text_threshold, pr.link_threshold, pr.low_text, pr.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]



        # get box + extend
        list_box = []
        for box in polys:
            [[l1, t1], [r1, t2], [r2, b1], [l2, b2]] = box
            if t1 < t2:
                l, r, t, b = l2, r1, t1, b1
            elif t1 > t2:
                l, r, t, b = l1, r2, t2, b2
            else:
                l, r, t, b = l1, r1, t1, b1

            xmin, ymin, xmax, ymax = l, t, r, b
            xmin, ymin, xmax, ymax = max(0, xmin - int((xmax - xmin) * pr.expand_ratio)),\
                                 max(0, ymin - int((ymax - ymin) * pr.expand_ratio)),\
                                 xmax + int((xmax - xmin) * pr.expand_ratio),\
                                 ymax + int((ymax - ymin) * pr.expand_ratio)
            list_box.append([xmin, ymin, xmax, ymax])

        # sort line
        dict_cum_sorted = self.sort_line_cmnd(list_box)
        list_box_optim = []
        for cum in dict_cum_sorted:
            for box in cum:
                list_box_optim.append(box)

        # draw box on image
        img_res = image.copy()
        img_res = np.ascontiguousarray(img_res)
        for box in list_box_optim:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(img_res, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (29, 187, 255), 2, 2)

        # crop image

        result_list_img_cum = []
        image_PIL = Image.fromarray(image)
        for cum in dict_cum_sorted:
            list_img = []
            for box in cum:
                xmin, ymin, xmax, ymax = box
                list_img.append(
                        image_PIL.copy().crop((xmin, ymin, xmax, ymax)))
            result_list_img_cum.append(list_img)
        return result_list_img_cum, img_res, None


    def sort_line_cmnd(self, boxes):

        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[1])   # sort by ymin
        lines = [[]]

        # y_center = (boxes[0][1] + boxes[0][3]) / 2.0
        y_max_base = boxes[0][3]    # y_max
        i = 0
        for box in boxes:
            if box[1] + 0.5 * abs(box[3] - box[1]) <= y_max_base:  # y_min <= y_max_base
                lines[i].append(box)
            else:
                lines[i] = sorted(lines[i], key=lambda x: x[0])
                # y_center = (box[1] + box[3]) / 2.0
                y_max_base = box[3]
                lines.append([])
                i += 1
                lines[i].append(box)

        temp = []

        for line in lines:
            temp.append(line[0][1])
        index_sort = np.argsort(np.array(temp)).tolist()
        lines_new = [self.remove(lines[i]) for i in index_sort]

        return lines_new
        # return lines

    def remove(self, line):
        line = sorted(line, key=lambda x: x[0])
        result = []
        check_index = -1
        for index in range(len(line)):
            if check_index == index:
                pass
            else:
                result.append(line[index])
                check_index = index
            if index == len(line) - 1:
                break
            if self.compute_iou(line[index], line[index + 1]) > 0.25:
                s1 = (line[index][2] - line[index][0] + 1) * (line[index][3] - line[index][1] + 1)
                s2 = (line[index + 1][2] - line[index + 1][0] + 1) * (line[index + 1][3] - line[index + 1][1] + 1)
                if s2 > s1:
                    del (result[-1])
                    result.append(line[index + 1])
                check_index = index + 1
        result = sorted(result, key=lambda x: x[0])
        return result

    def compute_iou(self, box1, box2):

        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        inter_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)

        s1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        s2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        # print(inter_area)
        iou = float(inter_area / (s1 + s2 - inter_area))

        return iou
    def sort_line(self, boxes):
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[1])
        lines = [[]]

        y_center = (boxes[0][1] + boxes[0][3]) / 2.0
        i = 0
        for box in boxes:
            if box[1] < y_center:
                lines[i].append(box)
            else:
                lines[i] = sorted(lines[i], key=lambda x: x[0])
                y_center = (box[1] + box[3]) / 2.0
                lines.append([])
                i += 1
                lines[i].append(box)

        temp = []

        for line in lines:
            temp.append(line[0][1])
        index_sort = np.argsort(np.array(temp)).tolist()
        lines_new = [self.remove(lines[i]) for i in index_sort]

        return lines_new


if __name__ == "__main__":
    app = CraftDetection()
    # app.test_folder("test_imgs")
    image = io.imread("test_imgs/meme.jpg")       # numpy array img (RGB order)
    image = np.array(image)
    boxes, img_res, _ = app.text_detect(image)
    plt.imshow(boxes[0])
    plt.show()
    plt.imshow(img_res)
    plt.show()
    # print(boxes)
    # print(score)

