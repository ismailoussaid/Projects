
import sys
import numpy as np
import pandas as pd
from math import floor
import cv2
from sklearn.model_selection import KFold
from tensorflow.keras.utils import Sequence, to_categorical

from platform_settings import *
from utils import *


class Augmentation:
    def __init__(self, other=None):
        self.other = other
    def process(self, img):
        if self.other is None:
            return img
        else:
            return self.other.process(img)


class Mirroring(Augmentation):
    def process(self, img):
        return cv2.flip(super().process(img), 1)


class Blurring(Augmentation):
    def process(self, img):
        return cv2.blur(super().process(img), (2, 2))


class MotionBlur(Augmentation):
    def __init__(self, blurtype, other=None):
        super().__init__(other)
         ## Build kernels for motion blur
        # The greater the size, the more the motion.
        kernel_size = 30
        # Create the vertical kernel.
        kernel = np.zeros((kernel_size, kernel_size))
        if blurtype == 'H':
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        elif blurtype == 'V':
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        else:
            raise NotImplementedError('Unknown blur type')
        # Normalize.
        kernel /= kernel_size
        self.kernel = kernel

    def process(self, img):
        return cv2.filter2D(super().process(img), -1, self.kernel)


class CelebASequence(Sequence):
    def __init__(self, attributes_path, images_path, shape, channel, max_items=None, n_split=5):
        self.images_path = images_path
        self.batch_size = 0
        self.sizes = (shape, shape, channel)
        self.kf = KFold(n_splits=n_split, shuffle=True)
        self.augmentations = [Augmentation()]  # non-augmented data always present

        self.attributes_tab = pd.read_csv(attributes_path)
        if max_items is not None:
            self.attributes_tab = self.attributes_tab.iloc[0:max_items]

        self.num_elements = len(self.attributes_tab['image_id'])
        print(f'Full Dataset has {self.num_elements} elements (before augmentation)')

    def augment(self, augmentation):
        self.augmentations.append(augmentation)

    def prepare(self, batch_size):
        self.batch_size = batch_size
        indexes = np.arange(0, self.num_elements * len(self.augmentations))
        self.input_train, self.input_test = [], []
        for train_index, test_index in self.kf.split(indexes):
            multiple_append([self.input_train, self.input_test], [train_index, test_index])

        print(f'Applied {len(self.augmentations)} augmentations, lead to {len(indexes)} elements')
        print(f'Each of the 5-Fold splits has: {len(self.input_train[0])} train and {len(self.input_test[0])} test elements')
        self.set_mode_train()
        self.set_mode_fold(0)

        self.attributes = ['Mustache', 'Eyeglasses', 'No_Beard', 'Wearing_Hat', 'Bald']
        self.attr_mapper = {'Mustache': 'mustache', 'Eyeglasses': 'eyeglasses', 'No_Beard': 'beard',
                            'Wearing_Hat': 'hat', 'Bald': 'bald'}

    def set_mode_train(self):
        self.mode = 0

    def set_mode_test(self):
        self.mode = 1

    def set_mode_fold(self, num_fold):
        self.fold = num_fold

    def __len__(self):
        if self.mode == 0:
            ln = len(self.input_train[self.fold])
        else:
            ln = len(self.input_test[self.fold])
        return floor(ln / self.batch_size)

    def __getitem__(self, idx):
        st, sp = int(idx * self.batch_size), int((idx + 1) * self.batch_size)

        imgs = np.empty((self.batch_size, *self.sizes))
        atts = {'mustache': [], 'eyeglasses': [], 'beard': [], 'hat': [], 'bald': []}

        j = 0
        for k in range(st, sp):
            if self.mode == 0:
                index = self.input_train[self.fold][k]
            else:
                index = self.input_test[self.fold][k]

            image_idx = int(floor(index / len(self.augmentations)))
            augm_idx = int(floor(index % len(self.augmentations)))

            image_name = self.attributes_tab['image_id'][image_idx]
            im = cv2.imread(self.images_path + image_name)
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
            img = self.augmentations[augm_idx].process(img)
            img = img.reshape(self.sizes)
            imgs[j, :, :, :] = img
            j += 1

            for a in self.attributes:
                name = self.attr_mapper[a]
                if name != 'beard':
                    atts[name].append(adapt(self.attributes_tab[a][image_idx]))
                else:
                    atts[name].append(anti_adapt(self.attributes_tab[a][image_idx]))

        out_attrs = {}
        for k, v in atts.items():
            out_attrs[k] = to_categorical(v, num_classes=2)

        return (imgs, out_attrs)



if __name__ == '__main__':

    def test():
        idem = Augmentation()
        m = Mirroring()
        b = Blurring()
        mb = Mirroring(Blurring())
        mmh = MotionBlur('H', Mirroring())

        size = 500
        test = np.diag(np.ones(size)) + np.diag(np.ones(size-1), 1) + np.diag(np.ones(size-1), -1)

        test0 = idem.process(test)
        test1 = m.process(test)
        test2 = b.process(test)
        test3 = mb.process(test)
        test4 = mmh.process(test)

        cv2.imshow('base', test)
        cv2.imshow('idem', test0)
        cv2.imshow('mirror', test1)
        cv2.imshow('blur', test2)
        cv2.imshow('mirror + blur', test3)
        cv2.imshow('horiz blur + mirror', test4)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_seq():
        batch_size = 64
        shape, channel = 36, 1
        max_items = 5000
        s = CelebASequence(attributes_path, images_path, shape, channel, max_items=max_items)
        s.augment(Mirroring())
        s.augment(Blurring())
        s.augment(Blurring(Mirroring()))
        s.augment(MotionBlur('H'))
        s.augment(MotionBlur('H', Mirroring()))
        s.prepare(batch_size)

        print(len(s))
        for i in s:
            print(i[0].shape)
            print(i[1].keys())
            break

    test()
    test_seq()
    sys.exit(0)
