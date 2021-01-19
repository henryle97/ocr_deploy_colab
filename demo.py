import time

import cv2
import streamlit as st
import SessionState
import joblib
from text_images import TEXT_IMAGES
import numpy as np
from PIL import Image
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/home/hisiter/IT/5_year/Graduation_Thesis /Generic_OCR/ocr_deploy/Object_Corner_Detection/center')
from detect import CENTER_MODEL


state = SessionState.get(result_text="", res="", prob_positive=0.0, prob_negative= 0.0, initial=True, img_drawed=None, img_cropped=None)

def main():
    model_text, model_cmnd_detect = load_model()
    st.title("Demo nhận dạng văn bản tiếng Việt")
    # Load model

    pages = {
        'Ảnh chế': page_meme,
        'CMND': page_cmnd

    }

    st.sidebar.title("Application")
    page = st.sidebar.radio("Chọn ứng dụng demo:", tuple(pages.keys()))

    pages[page](state, model_text, model_cmnd_detect)

    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model():
    print("Loading model ...")
    model_text = TEXT_IMAGES()
    model_cmnd_detect = CENTER_MODEL("./center/config/cmnd.yml")
    return model_text, model_cmnd_detect


def page_meme(state, model_text, model_cmnd_detect):
    st.header("Nhận dạng văn bản từ ảnh chế")

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        pil_image = Image.open(img_file_buffer)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # print(cv_image.shape)
        result_text, img_drawed_box, text_boxes, text_cropped_img = model_text.get_content_image(cv_image)
        state.result_text = result_text
        state.img_drawed = img_drawed_box
        state.img_cropped = text_cropped_img

    col1, col2= st.beta_columns(2)
    with col2:

        if state.result_text != "":
            result_text_format = []
            for texts in state.result_text:
                result_text_format.append(" ".join(texts))
            st.json(result_text_format)
    with col1:
        if state.img_drawed is not None:
            st.image(state.img_drawed, use_column_width=True)


    if state.img_cropped is not None:
        st.title("Chi tiết:")
        for idx, img in enumerate(state.img_cropped):
            st.image(img, caption=state.result_text[idx])
            st.empty()

def page_cmnd(state, model_text, model_cmnd_detect):
    st.header("Nhận dạng văn bản từ CMND")

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        pil_image = Image.open(img_file_buffer)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # print(cv_image.shape)

        # CMND detection
        t1 = time.time()
        cmnd_img, have_cmnd = model_cmnd_detect.detect_obj(cv_image)
        cmnd_detect_time = round(time.time() - t1, 2)
        print("CMND detect time: ", cmnd_detect_time)

        result_text, img_drawed_box, text_boxes, text_cropped_img = model_text.get_content_image(cmnd_img, have_cmnd, use_craft=True)
        state.result_text = result_text
        state.img_drawed = img_drawed_box
        state.img_cropped = text_cropped_img

        col1, col2 = st.beta_columns(2)
        with col2:

            if state.result_text != "":
                result_text_format = []
                for texts in state.result_text:
                    result_text_format.append(" ".join(texts))
                st.json(result_text_format)
        with col1:
            if state.img_drawed is not None:
                st.image(state.img_drawed, use_column_width=True)

        if state.img_cropped is not None:
            st.title("Chi tiết:")
            for idx, img in enumerate(state.img_cropped):
                st.image(img, caption=state.result_text[idx])
                st.empty()

if __name__ == "__main__":
    main()