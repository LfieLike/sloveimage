import os
import cv2
import numpy as np
import math
import sys, getopt
import torch
from pytorch_msssim import ssim,ms_ssim,SSIM,MS_SSIM
def getbpp(gt,dst):
    fsize = int(os.path.getsize(dst))*8
    #print(fsize)
    dpp = fsize/(gt.size*1.0)*3
    return dpp
def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def mse(img1,img2):
	return np.mean( (img1/255. - img2/255.) ** 2 )
# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
# 遍历文件夹
def my_sm_ssim(t_im1,t_im2):
    t_im1 = np.array(t_im1,dtype=np.float32)
    t_im2 = np.array(t_im2,dtype=np.float32)
    t_im1 = torch.from_numpy(t_im1).permute(0,3,1,2)
    t_im2 = torch.from_numpy(t_im2).permute(0,3,1,2)
    print(t_im1.shape)
    t_1 = t_im1.cuda()
    t_2 = t_im2.cuda()
    t_ssim =ms_ssim( t_1, t_2, data_range=255, size_average=False ).sum()
    return t_ssim
def walkFile(file,BPG,outs):
    print(file)
    print(BPG)
    print(outs)
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # # 遍历文件
        # for f in files:
        #     print(os.path.join(root, f))

        # 遍历所有的文件夹
        if(len(dirs)==0):
            continue
        index = 0
        t_psnr=0
        t_mse=0
        t_ssim=0
        t_bpp = 0
        t_im1=[]
        t_im2=[]
        for d in dirs:
            if(d[0]!= 'n'):
                continue
            #os.mkdir("/mnt/imagenet_data/BPG/val/"+d)
            for root_c,dirs_c,files_c in os.walk(os.path.join(root, d)):
                for f_c in files_c:
                    #print(f_c)
                    index+=1
                    
                    name1 = BPG+"/"+d+"/"+str(f_c)[:-4]+".png"
                    name2 = file+"/"+d+"/"+str(f_c)[:-4]+".png"
                    # print(name1)
                    # print(name2)
                    im1 = cv2.imread(name1)
                    im2 = cv2.imread(name2)
                    t_im1.append(im1)
                    t_im2.append(im2)
                    if im1.shape[0]!=im2.shape[0]:
                        im1 = im1.transpose(1,0,2)
                    #print(psnr2(im1,im2))
                    t_psnr = t_psnr+psnr2(im1,im2)
                    t_mse = t_mse + mse(im1,im2)
                    t_bpp = t_bpp + getbpp(im1,BPG+"/"+d+"/"+str(f_c)[:-4]+".myjpeg")
                    if(index % 500 == 0):
                        print("!!!!")
                        print(index)
                        print(str(index/50000.0)+"%")
                        print(name1)
                        print("t_psnr"+str(t_psnr/index))
                        t_ssim =t_ssim + my_sm_ssim(t_im1,t_im2)
                        t_im2 = []
                        t_im1 = []
                        print("t_ssim"+str(t_ssim/index))
                        print("t_mse"+str(t_mse/index))
                        print("t_bpp"+str(t_bpp/index))
        t_ssim/=index
        t_mse /=index
        t_bpp /=index
        t_psnr/=index
        print(BPG)
        print("psnr"+str(t_psnr))
        print("ssim"+str(t_ssim))
        print("mse"+str(t_mse))
        print("bpp"+str(t_bpp))
        print(outs+'.txt')
        with open(outs+'.txt', 'w') as f:
            f.write("bpp:"+str(t_bpp)+"\n")
            f.write("psnr:"+str(t_psnr)+"\n")
            f.write("ssim:"+str(t_ssim)+"\n")
            f.write("mse:"+str(t_mse)+"\n")
def main(argv):
    #walkFile("/mnt/imagenet_data/PNG/val","/mnt/imagenet_data/BPGtest/val","n01440764",1)
    inputfile = ''
    outputfile = ''
    quant = 0
    outs = ""
    try:
        opts, args = getopt.getopt(argv,"r:i:o:",["rfile","ifile=","ofile="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-r","--rfile"):
            outs = arg
    walkFile(inputfile,outputfile,outs)

if __name__ == "__main__":
   main(sys.argv[1:])
