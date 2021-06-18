import os
dir=[1,10,20,30,40,50,60,70,80,90,99]
for num in dir:
    print('python validate.py /home/yangxv/imagenet/j2kq'+str(num)+" --model resnet50 --pretrained")
    os.system('python validate.py /home/yangxv/imagenet/j2kq'+str(num)+" --model resnet50 --pretrained")
dir=[1,10,20,30,40,50,60,70,80,90,99]
for num in dir:
    print('python validate.py /home/yangxv/imagenet/jpg_q'+str(num)+" --model resnet50 --pretrained")
    os.system('python validate.py /home/yangxv/imagenet/jpg_q'+str(num)+" --model resnet50 --pretrained")
# import os
# dir=[1,10,20,30,35,40,45,50]
# for num in dir:
#     print('python validate.py /home/yangxv/imagenet/bpg_q'+str(num)+" --model resnet50 --pretrained")
#     os.system('python validate.py /home/yangxv/imagenet/bpg_q'+str(num)+" --model resnet50 --pretrained")