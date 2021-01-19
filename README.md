
## 1. Set up environment and dependent packages
+ Ubuntu 18.04
- Step 1: Set up conda environment (require conda is installed, if not, [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)):
```bash
conda env create -f environment.yml
conda activate py37_torch1.5
```
- Step 2: Build DCNv2:

```bash
cd BOX_MODEL/models/networks/DCNv2
sh ./make.sh
```
**Note** 

    * Build successfully if the output has line ***Zero offset passed*** or ***Passed gradient***
    * If there is any error, try downgrade cudatoolkit  

## 4. Testing:
- Download pretrained model:  https://drive.google.com/drive/folders/1-AVZ4Zd9jcJv5ZcqRfau7iClJ1Xa-WTe?usp=sharing
Đặt weight tại các đường dẫn sau: 
+ Object_Corner_Detection/weights/model_cmnd_best.pth
+ weights/vgg-seq2seq.pth
+ weights/weights_box.pth
```bash
python demo.py
```
