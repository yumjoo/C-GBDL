# coding=gbk
import os

import numpy as np
import torch
import torch.utils.data
import cv2
import random
from PIL import Image

fixed_img = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_txt, img_ext, mask_ext, transform=None, semi_setting=True, label_factor_semi=0.2,
                 rotate_flip=True, depth=96, crop_hw=96, random_whd_crop=True, num_classes=2, num_reference=8):

        self.data_txt = data_txt
        self.transform = transform
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.patient = []
        self.random_whd_crop = random_whd_crop
        self.depth = depth
        self.crop_hw = crop_hw
        self.semi_setting = semi_setting
        self.patient_label = []
        self.patient_unlabel = []
        self.rotate_flip = rotate_flip
        self.num_classes = num_classes
        self.num_reference = num_reference
        w = open(self.data_txt)
        # store images and label maps
        for ele in w.readlines():
            id = ele.split(' ')[0]
            id_l = ele.split(' ')[1].split('\n')[0]
            self.patient.append([id, id_l])

        num = len(self.patient)
        random.shuffle(self.patient)

        global fixed_img

        if self.semi_setting:
            num_patient = int(num * label_factor_semi)
            for i in range(num):
                if i < num_patient:
                    self.patient_label.append(self.patient[i])
                else:
                    self.patient_unlabel.append(self.patient[i])

            fixed_img = self.patient_label
            num_ratio = int(len(self.patient_unlabel) / len(self.patient_label))
            # copy to make they have the same length for sampling

            # balanced labeled and unlabeled data
            patient_label_repeat = self.patient_label * (num_ratio + 1)
            self.patient = patient_label_repeat + self.patient_unlabel
        else:
            # self.patient_label = fixed_img
            for i in range(num):
                self.patient_label.append(self.patient[i])

    def __len__(self):
        return len(self.patient)

    def rotate(self, image, label, angle, center=None, scale=1.0):

        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        rotated_label = cv2.warpAffine(label, M, (w, h))

        return rotated_image, rotated_label

    def random_crop(self, image, mask, crop_sz):
        img_sz = image.shape[0]
        random_arr = np.random.randint(img_sz - crop_sz + 1, size=2)
        y = random_arr[1]
        x = random_arr[0]
        h = crop_sz
        image_crop = image[y:y + h, x:x + h, :]
        mask_crop = mask[y:y + h, x:x + h, :]

        return image_crop, mask_crop

    def transform_crop(self, image, label, minx, maxx, miny, maxy, output_size=[112, 112]):

        w, h = label.shape

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2

        minx = max(minx - 10 - px, 0)
        maxx = min(maxx + 10 + px, w)
        miny = max(miny - 10 - py, 0)
        maxy = min(maxy + 10 + py, h)

        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy, :]
        label = label[minx:maxx, miny:maxy]

        label = np.expand_dims(label, -1)

        return image, label

    def __getitem__(self, idx):
        
        patient = self.patient[idx]

        imgs = []
        masks = []
        img_list = os.listdir(patient[0])

        num = len(img_list)
        base_name = patient[0].split('/')[-1]
        base_name_l = patient[1].split('/')[-1].split('\n')[0]

        nums = []

        image_shape = None
        mask_shape = None

        for ele in img_list:
            ele1 = ele.split('-')[-1]
            num_ = int(ele1.split('.')[0])
            nums.append(num_)

        base_num = min(nums)
        img_channel = 3
        img_paths = []
        num_collect = 0
        
        for i in range(0, num):
            ele = base_name + '-' + str(base_num + i) + '.' + self.img_ext
            ele_l = base_name_l + '-' + str(base_num + i) + '.' + self.mask_ext
            img = os.path.join(patient[0], ele)
            img_ = Image.open(img).convert('RGB')

            anno = os.path.join(patient[1], ele_l)
            mask_ = Image.open(anno).convert('L')

            img_sum = np.sum(np.array(img_).astype('float32'))
            mask_sum = np.sum(np.array(mask_).astype('float32'))

            if img_sum == 0.0 or mask_sum < 1.0:
                continue

            imgs.append(img_)
            masks.append(mask_)
            img_paths.append(img)
            num_collect += 1
        
        for i in range(len(imgs)):
            img_ = imgs[i]
            mask_ = masks[i]

            if self.transform is not None:
                imgs[i] = np.array(self.transform(img_))
                mask_ = np.expand_dims(np.array(self.transform(mask_)), -1)

            if np.max(mask_.astype('float32')) != 0.0:
                masks[i] = mask_.astype('float32') / np.max(mask_.astype('float32')) * (self.num_classes - 1)
            else:
                masks[i] = mask_
        
        img_mean = np.mean(imgs)
        img_std = np.std(imgs)

        for i in range(len(imgs)):
            if np.max(imgs[i].astype('float32')) > 0.0:
                imgs[i] = (imgs[i].astype('float32') - img_mean) / img_std

        image_shape = imgs[0].shape
        mask_shape = masks[0].shape

        if self.random_whd_crop:
            n_frame = len(imgs)
            if n_frame >= int(self.depth):
                start = random.randint(0, n_frame - self.depth)
                imgs = imgs[start:start + self.depth]
                masks = masks[start:start + self.depth]
                img_paths = img_paths[start:start + self.depth]
            else:
                N = self.depth - n_frame
                zero_img = np.zeros(image_shape)
                zero_mask = np.zeros(mask_shape)
                for i in range(N):
                    imgs.append(zero_img)
                    masks.append(zero_mask)
                    img_paths.append(' ')

        img = np.array(np.dstack(imgs))
        mask = np.array(np.dstack(masks))
        img_paths = np.dstack(img_paths)
    
        try:
            if self.random_whd_crop:
                img, mask = self.random_crop(img, mask, self.crop_hw)

            if self.rotate_flip:
                a = np.random.randint(2)
                if a == 0:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)

                b = np.random.randint(2)
                if b == 0:
                    angle = np.random.uniform(-10, 10)
                    img, mask = self.rotate(img, mask, angle)

        finally:
            
            img = img.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)

            img = np.expand_dims(img, 1)
            img = np.reshape(img, (-1, img_channel, img.shape[-2], img.shape[-1]))

            mask = np.expand_dims(mask, 1)
            mask = np.reshape(mask, (-1, 1, mask.shape[-2], mask.shape[-1]))

            mask_ones = np.ones(mask.shape)

            mask_one_hot = None
    
            for i in range(self.num_classes):
                if mask_one_hot is None:
                    mask_one_hot = (mask == ((i) * mask_ones)).astype('int')
                else:
                    mask_one_hot = np.concatenate((mask_one_hot, (mask == (i * mask_ones)).astype('int')), axis=1)
            
        # -----------------------------
        
        fixed_imgs_ = []
        fixed_mask_one_hots_ = []

        num_label = len(self.patient_label)

        for i in range(self.num_reference):
            idl = random.randint(0, num_label - 1)
            fixed_patient = self.patient_label[idl]

            fixed_imgs = []
            fixed_masks = []
            fixed_img_list = os.listdir(fixed_patient[0])

            fixed_num = len(fixed_img_list)
            fixed_name = fixed_patient[0].split('/')[-1]
            fixed_name_l = fixed_patient[1].split('/')[-1].split('\n')[0]

            fixed_nums = []
            for ele in fixed_img_list:
                ele1 = ele.split('-')[-1]
                num_ = int(ele1.split('.')[0])
                fixed_nums.append(num_)

            fixed_base_num = min(fixed_nums)
            fixed_img_paths = []
            fixed_num_collect = 0

            for i in range(0, fixed_num):
                ele = fixed_name + '-' + str(fixed_base_num + i) + '.' + self.img_ext
                ele_l = fixed_name_l + '-' + str(fixed_base_num + i) + '.' + self.mask_ext
                img0 = os.path.join(fixed_patient[0], ele)
                img_ = Image.open(img0).convert('RGB')

                anno = os.path.join(fixed_patient[1], ele_l)
                mask_ = Image.open(anno).convert('L')

                img_sum = np.sum(np.array(img_).astype('float32'))
                mask_sum = np.sum(np.array(mask_).astype('float32'))

                if img_sum == 0.0 or mask_sum < 1.0:
                    continue

                fixed_imgs.append(img_)
                fixed_masks.append(mask_)
                fixed_img_paths.append(img0)
                fixed_num_collect += 1

            for i in range(len(fixed_imgs)):
                img_ = fixed_imgs[i]
                mask_ = fixed_masks[i]

                if self.transform is not None:
                    fixed_imgs[i] = np.array(self.transform(img_))
                    mask_ = np.expand_dims(np.array(self.transform(mask_)), -1)

                if np.max(mask_.astype('float32')) != 0.0:
                    fixed_masks[i] = mask_.astype('float32') / np.max(mask_.astype('float32')) * (self.num_classes - 1)
                else:
                    fixed_masks[i] = mask_

            img_mean = np.mean(fixed_imgs)
            img_std = np.std(fixed_imgs)

            for i in range(len(fixed_imgs)):
                if np.max(fixed_imgs[i].astype('float32')) > 0.0:
                    fixed_imgs[i] = (fixed_imgs[i].astype('float32') - img_mean) / img_std

            image_shape = fixed_imgs[0].shape
            mask_shape = fixed_masks[0].shape

            if self.random_whd_crop:
                n_frame = len(fixed_imgs)
                if n_frame >= int(self.depth):
                    start = random.randint(0, n_frame - self.depth)
                    fixed_imgs = fixed_imgs[start:start + self.depth]
                    fixed_masks = fixed_masks[start:start + self.depth]
                    fixed_img_paths = fixed_img_paths[start:start + self.depth]
                else:
                    N = self.depth - n_frame
                    zero_img = np.zeros(image_shape)
                    zero_mask = np.zeros(mask_shape)
                    for i in range(N):
                        fixed_imgs.append(zero_img)
                        fixed_masks.append(zero_mask)
                        fixed_img_paths.append(' ')

            fixed_img = np.array(np.dstack(fixed_imgs))
            fixed_mask = np.array(np.dstack(fixed_masks))
            fixed_img_paths = np.dstack(fixed_img_paths)

            try:
                if self.random_whd_crop:
                    fixed_img, fixed_mask = self.random_crop(fixed_img, fixed_mask, self.crop_hw)

                if self.rotate_flip:
                    a = np.random.randint(2)
                    if a == 0:
                        fixed_img = cv2.flip(fixed_img, 1)
                        fixed_mask = cv2.flip(fixed_mask, 1)

                    b = np.random.randint(2)
                    if b == 0:
                        angle = np.random.uniform(-10, 10)
                        fixed_img, fixed_mask = self.rotate(fixed_img, fixed_mask, angle)

            finally:
                fixed_img = fixed_img.transpose(2, 0, 1)
                fixed_mask = fixed_mask.transpose(2, 0, 1)

                fixed_img = np.expand_dims(fixed_img, 1)
                fixed_img = np.reshape(fixed_img, (-1, img_channel, fixed_img.shape[-2], fixed_img.shape[-1]))

                fixed_mask = np.expand_dims(fixed_mask, 1)
                fixed_mask = np.reshape(fixed_mask, (-1, 1, fixed_mask.shape[-2], fixed_mask.shape[-1]))

                fixed_mask_ones = np.ones(fixed_mask.shape)

                fixed_mask_one_hot = None

                for i in range(self.num_classes):
                    if fixed_mask_one_hot is None:
                        fixed_mask_one_hot = (fixed_mask == ((i) * fixed_mask_ones)).astype('int')
                    else:
                        fixed_mask_one_hot = np.concatenate(
                            (fixed_mask_one_hot, (fixed_mask == (i * fixed_mask_ones)).astype('int')), axis=1)
            
            fixed_imgs_.append(fixed_img)
            fixed_mask_one_hots_.append(fixed_mask_one_hot)

        return img, mask_one_hot, num_collect, img_paths, self.patient_label, fixed_imgs_, fixed_mask_one_hots_

