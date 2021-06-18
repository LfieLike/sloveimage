import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# bpp
hyper_bpp = np.array([0.23, 0.35, 0.53, 0.79, 1.07, 1.56, 2.08, 2.67])
bpg_bpp = np.array([0.164, 0.362, 0.709, 1.286])
j2k_bpp = np.array([0.1445, 0.25, 0.429, 0.584, 0.723, 0.847, 0.989])
ours_bpp = np.array([0.658])

# pnsr
hyper_psnr = np.array([26.34, 27.84, 29.43, 31.13, 32.52, 34.71, 36.3, 37.93])
bpg_psnr = np.array([25.8, 28.56, 31.66, 34.99])
j2k_psnr= np.array([23.19, 25.22, 27.04, 28.22, 29.15, 29.9, 30.66])

ours_psnr = np.array([29.41])
# ours_psnr = 10. * np.log10(255.**2/ours_psnr)

# acc1
hyper_resnet18_acc1 = np.array([41.06, 49.73, 57.21, 62.3, 66.08, 67.77, 68.58, 69.31])
hyper_resnet18_acc1_org = np.ones_like(hyper_resnet18_acc1) * 69.55
hyper_resnet50_acc1 = np.array([46.01, 55.46, 62.63, 68.03, 71.85, 73.67, 74.85, 75.39])
hyper_resnet50_acc1_org = np.ones_like(hyper_resnet50_acc1) * 75.86
hyper_deit_T_acc1 = np.array([40.44, 49.21, 56.5, 62.14, 66.43, 68.79, 70.1, 71.01])
hyper_deit_T_acc1_org = np.ones_like(hyper_deit_T_acc1) * 72.13

bpg_resnet50_acc1 = np.array([47.14, 62.21, 70.43, 73.96])

j2k_resnet50_acc1 = np.array([27.47, 46.66, 59.56, 64.79, 67.63, 69.41, 70.87])

ours_acc1 = np.array([73.31])

# ms-ssim
hyper_msssim = np.array([0.9335, 0.9539, 0.9686, 0.9791, 0.9855, 0.9905, 0.9933, 0.9955])
bpg_msssim = np.array([0.9143, 0.9547, 0.9772, 0.9888])
j2k_msssim = np.array([0.847, 0.912, 0.949, 0.9636, 0.9712, 0.9759, 0.9799])
ours_msssim = np.array([0.969])
def load_xls(src):
    #return src excle data list 
    #list[map[]] 
    sheet = pd.read_excel(io=src)
    sheet_class = sheet.columns.to_list()
    ans = []
    for value in sheet.values:
        s_dict = dict()
        for i in range(len(value)):
            s_dict[sheet_class[i]]=value[i]
        ans.append(s_dict)
    return ans
def load_data():
    ans = dict()
    for root, dirs, files in os.walk("."):
        for file in files:
            if(file.endswith(".xls") == False):
                continue
            print(file)
            ans[file[:-4]]=load_xls(file)
    return ans
def get_data(str):
    sheets = load_data()
    ans_dict = dict()
    for fir in sheets:
        sec = sheets[fir]
        cnt = []
        for col in sec:
            cnt.append(col[str])
        ans_dict[fir] = cnt
    return ans_dict
def draw_bpp_acc():
    # acc1
    plt.subplots()
    plt.plot(hyper_bpp, hyper_resnet18_acc1, label="hyper+resnet18")
    plt.plot(hyper_bpp, hyper_resnet50_acc1, label="hyper+resnet50")
    plt.plot(hyper_bpp, hyper_deit_T_acc1, label="hyper+deit-T")
    plt.plot(ours_bpp, ours_acc1, marker="s", color="r", label="Ours (deit-T backbone)")
    acc1 = get_data('acc1')
    bpp = get_data('bpp')
    for name in acc1:
         plt.plot(bpp[name], acc1[name], label=name)
    # plt.plot(hyper_bpp, resnet50_acc1_org, linestyle="--", label="resnet50 upperbound")
    # plt.plot(hyper_bpp, deit_T_acc1_org, linestyle="--", label="deit-T upperbound")
    # plt.plot(bpg_bpp, bpg_resnet50_acc1, label="BPG+resnet50")
    # plt.plot(j2k_bpp, j2k_resnet50_acc1, label="J2K+resnet50")
    plt.xlabel('Bpp')
    plt.ylabel('Acc1(%)')
    plt.legend()
    plt.grid()
    plt.savefig('bpp_acc')
def draw_bpp_psnr():
    # psnr
    plt.subplots()
    plt.plot(hyper_bpp, hyper_psnr, label="hyper")
    # plt.plot(bpg_bpp, bpg_psnr, label="BPG")
    # plt.plot(j2k_bpp, j2k_psnr, label="j2k")
    plt.plot(ours_bpp, ours_psnr, marker="s", color="r", label="Ours (deit-T backbone)")
    bpp = get_data('bpp')
    psnr = get_data('PSNR')
    for name in bpp:
        plt.plot(bpp[name], psnr[name], label=name)
    plt.xlabel('Bpp')
    plt.ylabel('PSNR(dB)')
    plt.legend()
    plt.grid()
    plt.savefig('bpp_psnr')
def draw_bpp_msssim():
    # ms-ssim
    plt.subplots()
    plt.plot(hyper_bpp, hyper_msssim, label="hyper")
    # plt.plot(bpg_bpp, bpg_msssim, label="BPG")
    # plt.plot(j2k_bpp, j2k_msssim, label="J2K")
    plt.plot(ours_bpp, ours_msssim, marker="s", color="r", label="Ours (deit-T backbone)")
    bpp = get_data('bpp')
    msssim = get_data('MS-SSIM')
    for name in bpp:
        plt.plot(bpp[name], msssim[name], label=name)
    plt.xlabel('Bpp')
    plt.ylabel('MS-SSIM')
    plt.legend()
    plt.grid()

    plt.savefig('bpp_msssim')
##################################################################





x=load_data()
draw_bpp_acc()
draw_bpp_psnr()
draw_bpp_msssim()
for y in x:
    print(x[y])