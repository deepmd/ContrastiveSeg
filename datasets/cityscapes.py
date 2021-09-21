import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class Cityscapes(Dataset):
    def __init__(self, root, split, transforms, img_transform, label_transform, debug=False):
        """:param root(string) – Root directory of dataset where directory leftImg8bit and gtFine or gtCoarse
         are located.
         :param split (string, optional) – The image split to use, train, test or val
         :param transforms (callable, optional) – A function/transform that takes input sample and its target
         as entry and returns a transformed version.
         :param img_transform (callable, optional) – A function/transform that takes in an image and returns a
         transformed version. E.g, transforms.RandomCrop
         :param label_transform (callable, optional) – A function/transform that takes in the target and transforms it.
         """
        # self.root = root
        self.split = split
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.transforms = transforms

        self.img_list, self.label_list, self.name_list = self.__list_dirs(root)
        self.label_id_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        if debug:
            self.img_list = self.img_list[:80]
            self.label_list = self.label_list[:80]
            self.name_list = self.label_list[:80]

    def __list_dirs(self, root_dir):
        img_list = list()
        label_list = list()
        name_list = list()

        image_dir = os.path.join(root_dir, 'leftImg8bit', self.split)
        label_dir = os.path.join(root_dir, 'gtFine', self.split)

        # files = sorted(os.listdir(image_dir))
        for city in sorted(os.listdir(image_dir)):
            current_image_dir = os.path.join(image_dir, city)
            current_label_dir = os.path.join(label_dir, city)
            for file_name in os.listdir(current_image_dir):
                image_name = '.'.join(file_name.split('.')[:-1])

                img_path = os.path.join(current_image_dir, file_name)

                label_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'gtFine_labelIds.png')
                label_path = os.path.join(current_label_dir, label_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    continue
                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        return img_list, label_list, name_list

    # @staticmethod
    # def tonp(img):
    #     if isinstance(img, Image.Image):
    #         img = np.array(img)
    #
    #     return img.astype(np.uint8)

    @staticmethod
    def cv2_read_image(image_path, mode='RGB'):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mode == 'RGB':
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif mode == 'BGR':
            return img_bgr

        elif mode == 'P':
            return np.array(Image.open(image_path).convert('P'))

        else:
            print('Not support mode {}'.format(mode))
            exit(1)

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i, class_id in enumerate(self.label_id_list):
            encoded_labelmap[labelmap == class_id] = i

        return encoded_labelmap

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = self.cv2_read_image(self.img_list[index], mode='BGR')
        labelmap = self.cv2_read_image(self.label_list[index], mode='P')
        imag_name = self.name_list[index]

        if self.label_id_list is not None:
            labelmap = self._encode_label(labelmap)

        if self.transforms is not None:
            img, labelmap = self.transforms(img, labelmap=labelmap)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        return dict(
            img=img,
            labelmap=labelmap,
            index=index,
            imag_name=imag_name
        )


