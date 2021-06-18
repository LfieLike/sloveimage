import concurrent.futures as cf
import os    
import time
import cv2
import math
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from timm.data.random_erasing import RandomErasing
import math
import sys, getopt
import threading
R = threading.Lock()
def get_transforms():
    mean=IMAGENET_DEFAULT_MEAN
    std=IMAGENET_DEFAULT_STD
    interpolation='bicubic'
    tfl = [
            transforms.Resize(256, _pil_interp(interpolation)),
            transforms.CenterCrop(224),
        ]
    return transforms.Compose(tfl)
def checkGray(chip):
    # chip_gray = cv2.cvtColor(chip,cv2.COLOR_BGR2GRAY)
    r,g,b = cv2.split(chip)
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    s_w, s_h = r.shape[:2]
    x = (r+b+g)/3
 
    area_s=s_w * s_h
    # x = chip_gray
    r_gray = abs(r-x)
    g_gray = abs(g-x)
    b_gray=  abs(b-x)
    r_sum = np.sum(r_gray)/area_s
    g_sum = np.sum(g_gray)/area_s
    b_sum = np.sum(b_gray)/area_s
    gray_degree = (r_sum+g_sum+b_sum)/3
    if gray_degree <10:
        return True,gray_degree
    else:
        return False,gray_degree

def walkFile(src_path,dst_path,dir,quantit):
    R.acquire()
    if os.path.exists(dst_path) == False:
        os.makedirs(dst_path)
    if os.path.exists(dst_path+"/"+dir) == False:
        os.mkdir(dst_path+"/"+dir)
    R.release()
    tfs = get_transforms()
    for root, dirs, files in os.walk(src_path+"/"+dir):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # # 遍历文件
        # for f in files:
        #     print(os.path.join(root, f))

        # 遍历所有的文件夹
        for file in files:
            src = src_path + "/"+dir+"/"+file
            dst = dst_path + "/"+dir+"/"+file[:-4]+"ppm"
            #print(src)
            im = cv2.imread(src,0)
            #im = Image.fromarray(im) 
            #x = np.array(tfs(im))
            #print(x)
            #cv2.imwrite(dst,x)
            #print(im.shape[2])
            #print(im.shape[2]==1)
            falg,_ = checkGray(im)
            if flag :
                print(src)
#walkFile("/mnt/imagenet_data/PNG/val","/mnt/imagenet_data/PNG_re/val", "n01440764",45)
def main(argv):
    #walkFile("/mnt/imagenet_data/PNG/val","/mnt/imagenet_data/BPGtest/val","n01440764",1)
    inputfile = ''
    outputfile = ''
    quant = 0
    try:
        opts, args = getopt.getopt(argv,"i:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    tp = cf.ThreadPoolExecutor(16) # 设置线程数16
    futures = []
    startTime = time.time()
    for root, dirs, files in os.walk(inputfile):
        print(len(dirs))
        for dir in dirs:
            #future = tp.submit(walkFile,"/mnt/imagenet_data/PNG/val","/mnt/imagenet_data/PNG_re/val", dir,45)
            future = tp.submit(walkFile,inputfile,outputfile, dir,0)
            futures.append(future)
    count = 0
    for future in cf.as_completed(futures):
        count += 1
        endTime = time.time()
        runTime = endTime-startTime
        print(runTime)
    tp.shutdown()
    os.system('pause')



if __name__ == "__main__":
   main(sys.argv[1:])