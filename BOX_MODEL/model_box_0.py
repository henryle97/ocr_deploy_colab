from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import uuid
import os
import torch
from BOX_MODEL.models.networks.pose_dla_dcn import get_pose_net, load_model, pre_process, ctdet_decode, post_process, \
    merge_outputs
import cv2
from PIL import Image
import numpy as np
import time

class BOX_MODEL(object):
    def __init__(self):
        self.num_layers = 34
        self.heads = {'hm': 1, 'wh': 2, 'reg': 2}
        self.head_conv = 256
        self.scale = 1.0
        self.threshold = 0.15
        self.num_classes = 1
        self.threshold_x = 10
        self.threshold_y = 10
        self.list_label = ['text']
        self.model = get_pose_net(num_layers=self.num_layers, heads=self.heads, head_conv=self.head_conv)
        self.model = load_model(self.model, "weights/weights_box.pth")
        # self.device = torch.device("cuda:1")
        # self.model.cuda()
        self.model.eval()

    def predict_box(self, img):
        image, meta = pre_process(img, self.scale)
        # image = image.cuda()
        with torch.no_grad():
            start = time.time()
            output = self.model(image)[-1]
            print(time.time() - start)
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            dets = ctdet_decode(hm, wh, reg=reg, K=2000)
        dets = post_process(dets, meta)
        detections = [dets]
        results = merge_outputs(detections)
        list_box = []
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] >= self.threshold:
                    xmin, ymin, xmax, ymax = max(int(bbox[0]), 0), max(0, int(bbox[1])), min(int(bbox[2]),
                                                                                             img.shape[1]), min(
                        int(bbox[3]), img.shape[0])
                    list_box.append([xmin, ymin, xmax, ymax])

        lines = self.sort_line(list_box)
        print(lines)
        dict_cum_sorted = [self.sort_line_all(f) for f in self.get_cum(lines)]
        result_list_img_cum = []
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for cum in dict_cum_sorted:
            list_img = []
            for box in cum:
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                list_img.append(
                    img.copy().crop((max(0, xmin - int((xmax - xmin) * 0.03)), max(0, ymin - int((ymax - ymin) * 0.03)),
                                     xmax + int((xmax - xmin) * 0.03), ymax + int((ymax - ymin) * 0.03))).convert('L'))
            result_list_img_cum.append(list_img)

        return result_list_img_cum, _


        # result_lines = self.sort_line(list_box)
        # result_list_img = []
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # for line in result_lines:
        #     list_img = []
        #     for box in line:
        #         xmin = box[0]
        #         ymin = box[1]
        #         xmax = box[2]
        #         ymax = box[3]
        #         list_img.append(
        #         img.copy().crop((max(0, xmin - int((xmax - xmin) * 0.03)), max(0, ymin - int((ymax - ymin) * 0.03)),
        #                          xmax + int((xmax - xmin) * 0.03), ymax + int((ymax - ymin) * 0.03))).convert('L'))
        #     result_list_img.append(list_img)

        # return result_list_img

    def get_cum(self, lines):
        list_box = []
        for line in lines:
            list_box.extend(line)

        list_box_new = []
        for box in list_box:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            width = box[2] - box[0]
            height = box[3] - box[1]
            list_box_new.append([x_min, y_min, width, height, x_max, y_min])

        dict = []
        print(len(list_box))
        for i in range(len(list_box_new)):
            dict.append([i])

        for index1, b1 in enumerate(list_box_new):
            for index2, b2 in enumerate(list_box_new):
                if index1 == index2:
                    continue
                total_width = b1[2] + b2[2]
                total_height = b1[3] + b2[3]
                sub_x = abs(b1[0] - b2[0])
                sub_y = abs(b1[1] - b2[1])

                if sub_x < total_width and sub_y < total_height and b1[4] > b2[0]:
                    dict[index1].append(index2)

        dict = [sorted(f) for f in dict]

        list_index = []

        list_result = []

        for index1, c1 in enumerate(dict):
            cum_k = c1
            if index1 in list_index:
                continue
            for index2, c2 in enumerate(dict):
                if index1 == index2:
                    continue
                for b in c2:
                    if b in c1:
                        cum_k.extend(c2)
                        list_index.append(index2)
                        break
            list_result.append(cum_k)
            list_index.append(index1)

        list_cum_new = [sorted(list(set(f))) for f in list_result]

        dict = []

        for index1, c1 in enumerate(list_cum_new):
            if c1 not in dict:
                dict.append(c1)

        list_index = []

        list_result = []

        for index1, c1 in enumerate(dict):
            cum_k = c1
            if index1 in list_index:
                continue
            for index2, c2 in enumerate(dict):
                if index1 == index2:
                    continue
                for b in c2:
                    if b in c1:
                        cum_k.extend(c2)
                        list_index.append(index2)
                        break
            list_result.append(cum_k)
            list_index.append(index1)

        result = [sorted(list(set(f))) for f in list_result]

        result_toado = []
        for element in result:
            add = []
            for index in element:
                add.append(list_box[index])
            result_toado.append(add)
        return result_toado

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

    def sort_line_all(self, boxes):
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
        result = []
        lines_new = [self.remove(lines[i]) for i in index_sort]
        for index, line in enumerate(lines_new):
            result.extend(self.remove(line))

        return result


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
