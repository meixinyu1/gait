import os
import glob
import sys
import numpy as np
import os.path as osp
import random
import csv
from PIL import Image

class OULP_bag(object):
    count = 0
    def __init__(self, mark=True, user="meixinyu"):
        # self.root = '/home/meixinyu/gait/data/dataset_GEI'
        if user == 'meixinyu':
            self.root = '/home/meixinyu/gait/data/dataset_GEI'
        elif user == "zhangx":
            self.root = '/home/zhangsx/gait/data/dataset_GEI'
        elif user == "loki":
            self.root = "/home/jky/loki/work/gait/data/dataset_GEI"
        self.tag = np.zeros(100000)
        self.num_train_pids = 29097
        # labels_path = osp.join('/home/meixinyu/gait/data', 'id_list.csv')
        if user == 'meixinyu':
            labels_path = osp.join('/home/meixinyu/gait/data', 'id_list.csv')
        elif user == "zhangsx":
            labels_path = osp.join('/home/zhangsx/gait/data', 'id_list.csv')
        elif user == "loki":
            labels_path = osp.join("/home/jky/loki/work/gait/data", "id_list.csv")
        self.mark = mark
        line = 0
        with open(labels_path) as f:
            reader = csv.reader(f)
            for row in reader:
                line += 1
                if line < 4:
                    continue
                for i in range(5, 11, 1):
                    if row[i] == '1':
                        self.tag[line - 3] = int(i - 5 + 1)

        train, test, val = self.__split_data()

        self.train = self.__process_data(train)
        self.test_probe, self.test_gallery = self.__process_dense_data(test)
        # self.val_probe, self.val_gallery = self.__process_dense_data(val)
        print(self.count)

    def read_img(self, img_path):
        """
        keep reading until succeed
        :param img_path:
        :return:
        """
        img = None
        got_img = False
        while not got_img:
            try:
                img = Image.open(img_path)
                # img.close()
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        return img

    def __split_data(self):
        train_num = 29097
        # test_num = 29102
        test_num = 2000
        ids_list = os.listdir(self.root)

        print("before filter the len(id's imgs) < 2, the number of id is %d" % (len(ids_list)))
        ids_list_filter = []

        for idx in range(len(ids_list)):
            # print(idx)
            if ids_list[idx][0] == '.':
                continue
            id_path = osp.join(self.root, ids_list[idx])
            id_list = os.listdir(id_path)
            id_list.sort()
            if len(id_list) < 2 or id_list[0][len(id_list[0]) - 5] != '1':
                continue
            ids_list_filter.append(ids_list[idx])
        print("after filter the len(id's imgs) < 2, the number of id is %d" % (len(ids_list_filter)))

        # random.shuffle(ids_list_filter)

        return ids_list_filter[0:train_num], ids_list_filter[train_num: train_num + test_num], \
               ids_list_filter[train_num + test_num:]

    def __process_dense_data(self, dataset):
        probe = []
        gallery = []
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            choose = np.random.permutation(len(img_names))
            choose_probe = img_paths[0]
            choose_gallery = img_paths[1]
            probe_bag = np.zeros(6)
            gallery_bag = np.zeros(6)
            if choose_probe[len(choose_probe) - 5] == '1' and self.tag[int(id_name)] > 0:
                probe_bag[int(self.tag[int(id_name)]-1)] = 1
            if choose_gallery[len(choose_gallery) - 5] == '1' and self.tag[int(id_name)] > 0:
                gallery_bag[int(self.tag[int(id_name)]-1)] = 1
            img = np.asarray(self.read_img(choose_probe)).reshape((1, 128, 88))
            probe.append((img, int(id_name), probe_bag))
            img = np.asarray(self.read_img(choose_gallery)).reshape((1, 128, 88))
            gallery.append((img, int(id_name), gallery_bag))
        print("loading test over")
        return probe, gallery

    def __process_data(self, dataset):
        tracklets = []
        count = 0
        for idx in range(len(dataset)):
            id_name = dataset[idx]
            img_names = os.listdir(osp.join(self.root, id_name))
            img_paths = [osp.join(self.root, id_name, img_name) for img_name in img_names]
            img_paths.sort()
            not_choose = np.random.randint(1, 3, 1)
            index = 0
            for img_path in img_paths:
                if len(img_paths) != 2 and index == not_choose:
                    continue
                index += 1
                bag = np.zeros(6)
                if img_path[len(img_path) - 5] == '1' and self.tag[int(id_name)] > 0:
                    bag[int(self.tag[int(id_name)] - 1)] = 1

                img = np.asarray(self.read_img(img_path)).reshape((1, 128, 88))
                tracklets.append((img, idx, bag))
        print("loading train over")
        return tracklets


def init_dataset(mark, user):
    return OULP_bag(mark, user)
