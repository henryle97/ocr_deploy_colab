
cuda = False
trained_model = "weights/craft_mlt_25k.pth"
result_folder = "./result/"
canvas_size = 768             # image size for inference : smaller image --> lower accuracy, default 768
text_threshold = 0.4          # text confidence threshold
low_text = 0.38               # text low-bound score: 0.32 is best for word: space xung quanh 1 word, cang cao thi space cang it
link_threshold = 0.4          # link confidence threshold: threshold 2 ky tu lien ket voi nhau thanh 1 tu, upping de tach duoc nhieu tu

# best: 0.5, 0.32, 0.4

# not important
mag_ratio = 1.5               # image magnification ratio
poly = False                  # enable polygon type
refine = False                # enable link refiner
refiner_model = "weights/craft_refiner_CTW1500.pth"


# other utils: custom
horizontal_mode = False              # only horizontal boxes
ratio_box_horizontal = 0.3          # if number of horizontal boxes is larger than 50%, use horizontal mode
expand_ratio = 0.02                 # expand top, bottom: x%
visualize = True
show_time = True
folder_test = False

# api
host = 'localhost'
port = 1915


#text_threshold = Certainity required for a something to be classified as a letter. The higher this value the clearer characters need to look. I'd recommend 0.5-0.6
#link_threshold = Amount of distance allowed between two characters for them to be seen as a single word. I recommend 0.1-0.5, however playing with this value for your own use case might be better
#low_text = Amount of boundary space around the letter/word when the coordinates are returned. The higher this value the less space. Upping this value also affects the link threshold of seeing words as one, but it can cut off unecessary borders around leters. Having this value too high can affect edges of letters, cutting them off and lowering accuracy in reading them. I'd recommend 0.3-0.4
#CRAFT have difficulties with detecting a single character.