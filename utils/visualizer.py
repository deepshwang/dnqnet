import torch
import pdb
from PIL import Image
from scipy import ndimage
import os
import numpy as np


def visualize_featurespace(model, dataset):

    idx_1 = dataset.img_path_list.index('./data/RotNIST/data/test-images/15580_2.jpg')
    idx_2 = dataset.img_path_list.index('./data/RotNIST/data/test-images/15581_2.jpg')
    idx_3 = dataset.img_path_list.index('./data/RotNIST/data/test-images/15582_2.jpg')
    idx_4 = dataset.img_path_list.index('./data/RotNIST/data/test-images/15583_2.jpg')
    idx_5 = dataset.img_path_list.index('./data/RotNIST/data/test-images/15584_2.jpg')

    idx_list = [idx_1, idx_2, idx_3, idx_4, idx_5]

    img_name = dataset.img_path_list[idx_1].split("/")[-1].split("_")[0]
    root_dir = "./images/featurespace_comparison/" + img_name

    for i, idx in enumerate(idx_list):
        img, _ = dataset[idx]

        img = img.unsqueeze(0)
        feats = model(img)
        feats = torch.squeeze(feats)
        feats = torch.split(feats, 1)
        for j, feat in enumerate(feats):
            feat = torch.squeeze(feat)
            feat = feat.detach().numpy()

            dir_name = root_dir + "/feature_spaces/"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            filename = str(j) + "thfeat_" + str(i) + "th_rot.png"
            try:
                im = Image.fromarray(feat, mode='L')
            except ValueError:
                pdb.set_trace()
            save_str = dir_name + filename
            im.save(save_str)



    # img = torch.squeeze(dataset[idx][0]).numpy()
    #
    #
    # numpy_images=[img]
    # tensor_images = [torch.from_numpy(img).unsqueeze(0).unsqueeze(0)]
    #
    # for angle in rot_angles:
    #     numpy_image = ndimage.rotate(img, angle, reshape=False, cval=-0.5)
    #     numpy_images.append(numpy_image)
    #     tensor_images.append(torch.from_numpy(numpy_image).unsqueeze(0).unsqueeze(0))
    #
    # for i, np_img in enumerate(numpy_images):
    #     pil_im = Image.fromarray(np.uint8(np_img*255), 'L')
    #     dirname = root_dir + "/original_images/"
    #     if not os.path.exists(dirname):
    #         os.mkdir(dirname)
    #     save_str = dirname + str(angle_list[i]) + "_rot_original.png"
    #     pil_im.save(save_str)
    #
    # for i, img in enumerate(tensor_images):
    #     feats = torch.squeeze(model(img))
    #     feats = torch.split(feats, 1)
    #     for j, feat in enumerate(feats):
    #         feat = torch.squeeze(feat)
    #         feat = feat.detach().numpy()
    #         # For feature spaces with rotations, revert back
    #         if i != 0:
    #             feat = ndimage.rotate(feat, 360-angle_list[i], reshape=False, cval=-0.5)
    #
    #         dir_name = root_dir + "/feature_spaces/"
    #         if not os.path.exists(dir_name):
    #             os.mkdir(dir_name)
    #         filename = str(j) + "thfeat_" + str(angle_list[i]) + "_rot.png"
    #         try:
    #             im = Image.fromarray(feat, mode='L')
    #         except ValueError:
    #             pdb.set_trace()
    #         save_str = dir_name + filename
    #         im.save(save_str)
    # #
    # #
    # for i, feat in enumerate(up_out_split):
    #     feat = feat/torch.max(feat)*255
    #     feat = torch.squeeze(feat)
    #     feat = feat.detach().numpy()
    #     im = Image.fromarray(feat, mode='L')
    #     save_str = "./images/featurespace_comparison/" + str(i) + "_up.png"
    #     im.save(save_str)
