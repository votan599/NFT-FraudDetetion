
## <div align="center">ReadMe</div>

Reference [CSDN - YOLOv5 Cat Detetion](https://blog.csdn.net/oJiWuXuan/article/details/107558286) I read this article in the evening to conduct this project 

## <div align="center">Dependency</div>
[YOLOv5](https://github.com/ultralytics/yolov5)

## <div align="center">Environments</div>
OS：windows10
IDE：Pycharm
python version：anaconda Pyhon3.8
pytorch version: torch 1.10.1  # PyTorch 1.11.0 has a issue can't display the result, 
cuda version: 11.3
GPU: NVIDIA Geforce GTX 960 (a 4 years ago hardware)

## <div align="center">Quick Start Examples</div>
<details open>
<summary>Environment Setup</summary>
  * Anaconda
  * [PyTorch](https://pytorch.org/get-started/previous-versions/)
  * [YOLOv5](https://github.com/ultralytics/yolov5)
    
```bash
# Conda Environment----------------------------------
conda create -n yolov5 python=3.8 conda activate yolov5

# PyTorch & CUDA Install -------------------------------------
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Base ----------------------------------------
pip install matplotlib
pip install numpy
pip install opencv-python
pip install pillow
pip install pyyaml
pip install requests
pip install scipy
pip install tqdm
pip install tensorboard
pip install pandas
pip install seaborn
pip install thop
pip install Cython
pip install pycocotools
```
</details>

<details open>
<summary>Detetion (using a well trained model)</summary>
  Store pics under .\data\final
 
```bash
  python detect.py --weights ./runs/train/exp3/weights/best.pt --img 640 --source ./data/final/ --save-txt --save-conf
```
</details>
<details open>
<summary>Retrain - model (if needed)</summary>
  * Put all required training pics under .\data\images (recommened more than 50 pics)
  * Put tag data under .\data\Annotations   I use [this software](http://www.jinglingbiaozhu.com/)
```bash
  # Divided pics as training,test and etc.----------------------------------------
  python makeTxt.py
  # Exact tag data from xml files under .\data\Annotations ----------------------------------------
  python voc_label.py
  # Start training;  epochs>=100 ----------------------------------------
  python train.py --img 640 --batch 50 --epochs 100 --data nft.yaml --weights yolov5s.pt
```
</details>
