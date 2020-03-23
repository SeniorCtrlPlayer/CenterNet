# -*- coding: utf-8 -*-
import _init_paths
import cv2
import numpy as np
import sys
# sys.path.append('../src')
from lib.datasets.multi_pose import MultiPoseDataset
from lib.datasets.coco_hp import COCOHP
from opts import opts
from torch.utils.data import DataLoader

def get_dataset():
  class Dataset(COCOHP, MultiPoseDataset):
    pass
  return Dataset

rows=0
cols=0

def get_dir(src_point, rot_rad):
    # point rot transformation - by lwk
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    # get transform_matrix [2x3]
    # scale: size of output while fix_res is True
    #        size of center_output while fix_res is False
    # uniform scale
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    # src_dir = get_dir([0, src_w * -0.5], rot_rad)
    src_dir = np.array([0, src_w * -0.5], np.float32)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def test(name):
    img_origin = cv2.imread(name)
    cv2.imshow('input', img_origin)
    #cv.waitKey(0)
    rows, cols = img_origin.shape[:2]
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[50,100],[200,100],[50,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    print(M)
    #第三个参数：变换后的图像大小
    res = cv2.warpAffine(img_origin,M,(rows//2,cols//2))
    cv2.imshow('output', res)
    if cv2.waitKey(1) == 27:
        return  # esc to quit
def test2(name):
    img_origin = cv2.imread(name)
    h, w = img_origin.shape[:2]
    c = np.array([w/2., h/2.])
    s = max(img_origin.shape[:2]) * 0.5
    trans = get_affine_transform(c, s, 0, [512, 512])
    inp = cv2.warpAffine(img_origin, trans, (512,512), flags=cv2.INTER_LINEAR)
    # print(inp.shape)
    cv2.imshow('input', inp)
    cv2.waitKey(27)
    # print(c)
def test3():
    Dataset = get_dataset()
    opt = opts().parse()
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Dataset = Dataset(opt, 'val')
    test_dataloader = DataLoader(Dataset,
                                 1,
                                 num_workers=1,
                                 shuffle=False)
    inp = 1
    heatmap = 1
    for idx, a in enumerate(test_dataloader):
        if a['reg_mask'][0].sum() > 6:
            origin = a['origin'][0].numpy()
            inp = a['input'][0].permute(1,2,0).numpy()
            heatmap = a['hm'][0].permute(1,2,0).numpy()
            hm_hp = a['hm_hp'][0][0].unsqueeze(dim=-1).numpy()
            break
    # print(inp.size())
    # print(hm_hp.size())
    # print(origin.shape)
    cv2.imshow('origin',origin)
    cv2.imshow('input',inp)
    cv2.imshow('hm', heatmap)
    cv2.imshow('hm_hp', hm_hp)
    cv2.waitKey(27)
def gaussian2D(shape, sigma=1):
    # 得到高斯方格
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
def test4():
    r=20
    a = gaussian2D((r,r), r/6.)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x=np.arange(0,r,1).reshape((1,r)).repeat(r, axis=0).reshape(r,r)
    y=np.arange(0,r,1).repeat(r, axis=0).reshape(r,r)
    fig=plt.figure()
    ax=Axes3D(fig)
    ax=plt.gca(projection='3d')
    ax.plot_surface(x,y,a,cmap='rainbow')
    plt.title('r=20 gaussian2D')
    plt.show()
    cv2.imshow('gaussian2D', a.reshape(r,r,1))
    cv2.waitKey(27)
# test2('16004479832_a748d55f21_k.jpg')
# src = np.zeros((3, 2), dtype=np.float32)
# center = np.array([rows // 2, cols // 2], dtype=np.float32)
# src[0, :] = center
# print(src)
if __name__=='__main__':
    test3()