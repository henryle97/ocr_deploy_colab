from PIL import Image
import cv2
from BOX_MODEL.model_box import BOX_MODEL
import time
import matplotlib.pyplot as plt
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np
from CRAFT_pytorch.predict_box import CraftDetection


class TEXT_IMAGES(object):

    def __init__(self, reg_model='seq2seq'):
        print("Loading TEXT_MODEL...")
        if reg_model == "seq2seq":
            config = Cfg.load_config_from_name('vgg_seq2seq')
            config['weights'] = 'weights/vgg-seq2seq.pth'

        self.model_box = BOX_MODEL()
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False

        self.model_reg = Predictor(config)
        self.craft_model = CraftDetection()




    def get_content_image(self, image, have_cmnd=False, use_craft=False):
        # cv image
        # return image_drawed, texts, boxes
        t1 = time.time()
        if use_craft:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text_cropped_img, img_drawed_box, text_boxes = self.craft_model.text_detect(image, have_cmnd)
        else:
            text_cropped_img, img_drawed_box, text_boxes = self.model_box.predict_box(image, have_cmnd)
        detect_text_time = round(time.time() - t1,2)
        print("Time detect text: ", detect_text_time)

        # print(text_boxes)


        t2 = time.time()

        result_text = []
        for cluster in text_cropped_img:

            text_with_size_batch, order_dict = self.clustering_text_by_width(cluster)
            text_predict_dict = {65: [], 128: []}
            text_predict_ordered = []
            for standard_size, cluster_new in text_with_size_batch.items():
                if len(cluster_new) == 0:
                    continue
                text_predict = self.model_reg.predict_batch(cluster_new, standard_size)
                text_predict_dict[standard_size] += text_predict

            idx = 0
            while idx < len(cluster):
                text_predict_ordered += [text_predict_dict[order_dict[idx]].pop(0)]
                idx += 1
            result_text.append(text_predict_ordered)

        reg_text_time = round(time.time() - t2, 2)
        print("Time recognize text: ", reg_text_time)
        return result_text, img_drawed_box, text_boxes, text_cropped_img

    def clustering_text_by_width(self, text_imgs):
        new_cluster = {65: [],
                       128: []}

        order_dict = {}

        for i, img in enumerate(text_imgs):
            w, h = img.size
            new_w = w / h * 32
            if new_w <= 65:
                new_cluster[65].append(img)
                order_dict[i] = 65
                i += 1
            else:
                new_cluster[128].append(img)
                order_dict[i] = 128
                i += 1

        return new_cluster, order_dict


if __name__ == "__main__":
    app = TEXT_IMAGES()
    # app.model_reg.quantization_model()
    img_path ="/home/hisiter/IT/5_year/Graduation_Thesis /Generic_OCR/ocr_deploy/image/cmnd/10134.jpg"
    img = cv2.imread(img_path)
    result_text, img_drawed_box, text_boxes, text_cropped_img = app.get_content_image(img, use_craft=True)
    print(result_text)
    plt.imshow(img_drawed_box)
    plt.show()
    # print(text_boxes)
    # print(res)
