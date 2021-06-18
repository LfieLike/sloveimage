import concurrent.futures as cf
import os    
import time
import sys,getopt
import threading
from PIL import Image
R = threading.Lock()
def walkFile(src_path,dst_path,dir,quantit):
    R.acquire()
    if os.path.exists(dst_path) == False:
        os.makedirs(dst_path)
    if os.path.exists(dst_path+"/"+dir) == False:
        os.mkdir(dst_path+"/"+dir)
        print(dir)
    R.release()
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
            dst = dst_path + "/"+dir+"/"+file[:-4]+".myjpeg"
            if os.path.exists(dst) == True:
                continue
            # os.system("/home/yangxv/kakadu/kdu_compress -o "+ dst+" Qfactor="+str(quantit)+" -i "+src )
            # os.system("/home/yangxv/kakadu/kdu_expand  -o "+ dst_path + "/"+dir+"/"+file[:-4]+".ppm"+" -i "+dst)
            im = Image.open(src)
            im.save(dst,format='JPEG',quality=quantit)
            im = Image.open(dst)
            im.save(dst_path + "/"+dir+"/"+file[:-4]+".png",format = 'PNG')

def main(argv):
    #walkFile("/mnt/imagenet_data/PNG/val", ,"n01440764",1)
    inputfile = ''
    outputfile = ''
    quant = 0
    try:
        opts, args = getopt.getopt(argv,"q:i:o:",["quant","ifile=","ofile="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-q","--quant"):
            quant = int(arg)
    tp = cf.ThreadPoolExecutor(16) # 设置线程数16
    futures = []
    startTime = time.time()
    print(inputfile)
    for root, dirs, files in os.walk(inputfile):
        print(len(dirs))
        for dir in dirs:
            future = tp.submit(walkFile,inputfile,outputfile, dir,quant)
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
