import os
import random
import torch
import torch.utils.data
from PIL import Image
import sys
import scipy.io as scio
import cv2
import numpy as np
from mosaic import mosaic,CLASSES
from lxml import etree
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList


class neu(torch.utils.data.Dataset):
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """
    CLASSES = ("__background__ ","crazing", "inclusion", "patches","pitted_surface","rolled-in_scale","scratches")

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, external_proposal=False, old_classes=[],
                 new_classes=[], excluded_classes=[], is_train=True):
        self.root = data_dir
        self.image_set = split  # train, validation, test
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.use_external_proposal = external_proposal

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        self._proposalpath = os.path.join( self.root, "EdgeBoxesProposals", "%s.mat")

        self._img_height = 0
        self._img_width = 0

        self.old_classes = old_classes
        self.new_classes = new_classes
        self.exclude_classes = excluded_classes
        self.is_train = is_train
        self.CTN = False
        # true代表只是用当前新类数据进行训练第二个老师

        # self.data_augmentation = False# True 是否使用mosaic数据增强方法

        # load data from all categories
        # self._normally_load_defect()

        # do not use old data
        if self.is_train:  # training mode

            print('neu.'
                  'py | in training mode')

            self._load_img_from_NEW_cls_without_old_data()

        else:
            print('neu.py | in test mode')
            self._load_img_from_NEW_and_OLD_cls()

    def _normally_load_defect(self):
        """ load data from all 20 categories """

        # print("voc.py | normally_load_voc | load data from all 20 categories")
        print("neu.py | normally_load_defect | load data")
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.final_ids = self.ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}  # image_index : image_id

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))  # class_name : class_id

    def _load_img_from_NEW_and_OLD_cls(self):
        self.ids = []
        if self.CTN == True:
            total_classes = self.new_classes
        else:
            total_classes = self.new_classes + self.old_classes
        for w in range(len(total_classes)):
            incremental = total_classes[w]
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}".format(self.image_set)) as f:
                # with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f:
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]

            for i in range(len(buff)):
                x = buff[i].split("\\")[-1]
                # x = buff[i]
                x = x.split(' ')
                x[0]=x[0][:-4]
                anno_path = self._annopath % "{0}".format(x[0])
                tree = etree.parse(anno_path)
                root = tree.getroot()
                objectes = root.findall('.//object')
                bnd_box = []

                for object in objectes:
                    name = object.find("name").text
                    if name == incremental and (x[0] not in img_ids_per_category):
                        img_ids_per_category.append(x[0])
                        self.ids.append(x[0])

                # if x[1] == '-1':
                #     pass
                # elif x[2] == '0':  # include difficult level object
                #     if self.is_train:
                #         pass
                #     else:
                #         img_ids_per_category.append(x[0])
                #         self.ids.append(x[0])
                # else:
                #     img_ids_per_category.append(x[0])
                #     self.ids.append(x[0])

            print('neu.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(
                incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            self.final_ids = []
            for id in self.ids:
                repeat_flag = False
                for final_id in self.final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.final_ids.append(id)


            # store image ids and class ids

        print(
            'neu.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(
                self.image_set, len(self.final_ids)))
        self.final_ids.sort()
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = neu.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def _load_img_from_NEW_cls_without_old_data(self):
        self.ids = []

        for incremental in self.new_classes:  # read corresponding class images from the data set
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}".format(self.image_set)) as f:
            # with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f:
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]

            for i in range(len(buff)):
                x = buff[i].split("\\")[-1]
                x = x.split(' ')
                x[0]=x[0][:-4]
                anno_path = self._annopath % "{0}".format(x[0])
                tree = etree.parse(anno_path)
                root = tree.getroot()
                objectes = root.findall('.//object')
                bnd_box = []

                for object in objectes:
                    name = object.find("name").text
                    if name == incremental and (x[0] not in img_ids_per_category):
                        img_ids_per_category.append(x[0])
                        self.ids.append(x[0])

                # if x[1] == '-1':
                #     pass
                # elif x[2] == '0':  # include difficult level object
                #     if self.is_train:
                #         pass
                #     else:
                #         img_ids_per_category.append(x[0])
                #         self.ids.append(x[0])
                # else:
                #     img_ids_per_category.append(x[0])
                #     self.ids.append(x[0])



            print('neu.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            self.final_ids = []
            for id in self.ids:
                repeat_flag = False
                for final_id in self.final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.final_ids.append(id)

        # store image ids and class ids
        print('neu.py | load_img_from_NEW_cls_without_old_data | total used number of images in {0}: {1}'.format(
            self.image_set, len(self.final_ids)))

        self.final_ids.sort()
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = neu.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def _load_img_from_old_cls_without_new_data(self):
        self.old_ids = []

        for incremental in self.old_classes:  # read corresponding class images from the data set
        # for incremental in self.new_classes:
            num = 0
            flag = False
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}".format(self.image_set)) as f:
            # with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f:
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]

            for i in range(len(buff)):
                if flag == True:
                    break
                x = buff[i]
                x = x.split(' ')
                anno_path = self._annopath% "{0}".format(x[0])
                tree = etree.parse(anno_path)
                root = tree.getroot()
                objectes = root.findall('.//object')
                bnd_box = []

                for object in objectes:
                    name = object.find("name").text
                    if name == incremental and (x[0] not in img_ids_per_category):
                        num = num+1
                        img_ids_per_category.append(x[0])
                        self.old_ids.append(x[0])
                        if num == 10 :
                            flag = True


                # if x[1] == '-1':
                #     pass
                # elif x[2] == '0':  # include difficult level object
                #     if self.is_train:
                #         pass
                #     else:
                #         img_ids_per_category.append(x[0])
                #         self.ids.append(x[0])
                # else:
                #     img_ids_per_category.append(x[0])
                #     self.ids.append(x[0])



            self.old_final_ids = self.final_ids
            print('neu.py | load_img_from_old_cls_without_new_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            self.old_final_ids = []
            for id in (self.old_ids+self.final_ids):
                repeat_flag = False
                for final_id in self.old_final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.old_final_ids.append(id)

        # store image ids and class ids
        self.old_final_ids = self.final_ids
        print('neu.py | load_img_from_old_cls_without_new_data | total used number of images in {0}: {1}'.format(
            self.image_set, len(self.old_final_ids)))

        self.old_final_ids.sort()
        self.old_id_to_img_map = {k: v for k, v in enumerate(self.old_final_ids)}
        cls = neu.CLASSES
        self.old_class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):


        img_id = self.final_ids[index]

        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)

        target = target.clip_to_image(remove_empty=True)

        if self.use_external_proposal:
            proposal = self.get_proposal(index)
            proposal = proposal.clip_to_image(remove_empty=True)
        else:
            proposal = None



        if self.transforms is not None:
            img, target, proposal = self.transforms(img, target, proposal)

        return img, target, proposal, img_id

    def __len__(self):
        return len(self.final_ids)

    def get_groundtruth(self, index):
        img_id = self.final_ids[index]

        # print(img_id)
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        img = Image.open(self._imgpath % "{0}".format(img_id)).convert("RGB")

        # anno = ET.parse(self._annopath % img_id).getroot()
        # size = anno.find("size")
        im_info = tuple(map(int, (img.size[1], img.size[0])))

        anno["im_info"] = im_info

        height, width = anno["im_info"]
        self._img_height = height
        self._img_width = width
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.clip_to_image(remove_empty=True,TO_REMOVE=None)
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def get_proposal(self, index):
        boxes = []

        img_id = self.final_ids[index]
        proposal_path = self._proposalpath % "{0}".format(img_id)
        proposal_raw_data = scio.loadmat(proposal_path)
        proposal_data = proposal_raw_data['bbs']
        proposal_length = proposal_data.shape[0]
        for i in range(2000):
            # print('i: {0}'.format(i))
            if i >= proposal_length:
                break
            left = proposal_data[i][0]
            top = proposal_data[i][1]
            width = proposal_data[i][2]
            height = proposal_data[i][3]
            score = proposal_data[i][4]
            right = left + width
            bottom = top + height
            box = [left, top, right, bottom]
            boxes.append(box)
        img_height = self._img_height
        img_width = self._img_width

        boxes = torch.tensor(boxes, dtype=torch.float32)
        proposal = BoxList(boxes, (img_width, img_height), mode="xyxy")

        return proposal

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        self.old_class_num = len(self.old_classes)
        for obj in target.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            difficult = 0
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()

            old_class_flag = False
            for old in self.old_classes:
                if name == old:
                    old_class_flag = True
                    break
            exclude_class_flag = False
            for exclude in self.exclude_classes:
                if name == exclude:
                    exclude_class_flag = True
                    break

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [bb.find("xmin").text, bb.find("ymin").text, bb.find("xmax").text, bb.find("ymax").text]
            bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

            if exclude_class_flag:
                pass
                # print('voc.py | incremental train | object category belongs to exclude categoires: {0}'.format(name))
            elif self.is_train and old_class_flag:
                pass
                # print('voc.py | incremental train | object category belongs to old categoires: {0}'.format(name))
            else:
                #增加对CNT的标签处理，
                boxes.append(bndbox)
                if self.class_to_ind[name]>self.old_class_num and self.CTN:
                    gt_classes.append(self.class_to_ind[name]-self.old_class_num)
                else:
                    gt_classes.append(self.class_to_ind[name])
                difficult_boxes.append(difficult)


        # size = target.find("size")
        # im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            # "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.final_ids[index]
        img = Image.open(self._imgpath % "{0}".format(img_id)).convert("RGB")

        # anno = ET.parse(self._annopath % img_id).getroot()
        # size = anno.find("size")
        im_info = tuple(map(int, (img.size[1], img.size[0])))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        if self.CTN == True:
            try:
                return neu.CLASSES[class_id+self.old_class_num]
            except:
                print("error")
        else:
            return neu.CLASSES[class_id ]

    def get_img_id(self, index):
        img_id = self.final_ids[index]
        return img_id

    def mosaic(self, img_id,imgsetpath):

        img_paths, annos = self.get_dataset(imgsetpath)
        # print('长度:' + str(len(img_paths)) + ';' + str(len(annos)))
        # print(len(img_paths))
        # print(len(annos))
        # print(type(img_paths))
        # print(type(annos))
        # print(img_paths)
        # print(annos)
        # set_trace()
        random.seed()
        idxs = random.sample(range(len(annos)), 3)  # 从annos列表长度中随机取3个数
        # print(idxs)
        idxs.append(self.old_final_ids.index(img_id))
        # print(idxs)
        random.shuffle(idxs)
        # print(idxs)
        # set_trace()
        OUTPUT_SIZE = (768 , 768 )
        SCALE_RANGE = (0.3, 0.7)
        FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.
        new_image, new_annos, gt_class = self.update_image_and_anno(img_paths, annos,
                                                               idxs,
                                                               OUTPUT_SIZE, SCALE_RANGE,
                                                               filter_scale=FILTER_TINY_SCALE)

        difficult = torch.zeros(len(new_annos),dtype=torch.int64)
        im_info = tuple(map(int, (new_image.shape[1], new_image.shape[0])))

        # for anno in new_annos:
        #     start_point = (int(anno[0] ), int(anno[1]))  # 左上角点
        #     end_point = (int(anno[2]), int(anno[3] ))  # 右下角点
        #     cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)  # 每循环一次在合成图画一个矩形
        #     cv2.putText(new_image, 'biaoqian', start_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0))
        # cv2.imshow("show",new_image)
        # cv2.waitKey(0)

        return new_image, {
            "boxes": torch.tensor(new_annos, dtype=torch.float32),
            "labels": torch.tensor(gt_class),
            "difficult": difficult,
            "im_info": im_info,
        }

        # # 更新获取新图和对应anno
        # cv2.imwrite(save_img + str(img_id) + '.jpg', new_image)
        # # # annos是
        # xml = GEN_Annotations(str(img_id) + '.jpg')
        # xml.set_size(768, 768, 3)
        # for anno in new_annos:
        #     start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))  # 左上角点
        #     end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))  # 右下角点
        #     cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)  # 每循环一次在合成图画一个矩形
        #     cv2.putText(new_image, 'biaoqian', start_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0))
        #     createXml(anno[0],start_point[0],start_point[1],end_point[0],end_point[1],xml)
        #     pass
        # xml.savefile(save_xml+str(img_id)+'.xml')
        # #
        # cv2.imwrite(save_img_bbox+str(img_id)+'.jpg', new_image)

        # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        # new_image = Image.fromarray(new_image.astype(np.uint8))
        # new_image.show()
        # cv2.imwrite('./img/wind_output111.jpg', new_image)

    def update_image_and_anno(self,all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
        OUTPUT_SIZE = output_size
        output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
        # scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        # scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        scale_x = 0.5
        scale_y = 0.5
        divid_point_x = int(scale_x * output_size[1])
        divid_point_y = int(scale_y * output_size[0])

        new_anno = []
        gt_class = []
        # print(idxs)
        for i, idx in enumerate(idxs):
            # set_trace()
            path = all_img_list[idx]
            img_annos = all_annos[idx]

            img = cv2.imread(path)
            if i == 0:  # top-left
                img = cv2.resize(img, (divid_point_x, divid_point_y),interpolation=cv2.INTER_AREA)
                output_img[:divid_point_y, :divid_point_x, :] = img
                for bbox in img_annos:
                    xmin = bbox[1] * scale_x
                    ymin = bbox[2] * scale_y
                    xmax = bbox[3] * scale_x
                    ymax = bbox[4] * scale_y
                    xmin, ymin, xmax, ymax = int(xmin * OUTPUT_SIZE[0]), int(ymin * OUTPUT_SIZE[1]), int(
                        xmax * OUTPUT_SIZE[0]), int(ymax * OUTPUT_SIZE[1])
                    new_anno.append([xmin, ymin, xmax, ymax])
                    gt_class.append(CLASSES[bbox[0]])
            elif i == 1:  # top-right
                img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y),interpolation=cv2.INTER_AREA)
                output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
                for bbox in img_annos:
                    xmin = scale_x + bbox[1] * (1 - scale_x)
                    ymin = bbox[2] * scale_y
                    xmax = scale_x + bbox[3] * (1 - scale_x)
                    ymax = bbox[4] * scale_y
                    xmin, ymin, xmax, ymax = int(xmin * OUTPUT_SIZE[0]), int(ymin * OUTPUT_SIZE[1]), int(
                        xmax * OUTPUT_SIZE[0]), int(ymax * OUTPUT_SIZE[1])
                    new_anno.append([xmin, ymin, xmax, ymax])
                    gt_class.append(CLASSES[bbox[0]])
            elif i == 2:  # bottom-left
                img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y),interpolation=cv2.INTER_AREA)
                output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
                for bbox in img_annos:
                    xmin = bbox[1] * scale_x
                    ymin = scale_y + bbox[2] * (1 - scale_y)
                    xmax = bbox[3] * scale_x
                    ymax = scale_y + bbox[4] * (1 - scale_y)
                    xmin, ymin, xmax, ymax = int(xmin * OUTPUT_SIZE[0]), int(ymin * OUTPUT_SIZE[1]), int(
                        xmax * OUTPUT_SIZE[0]), int(ymax * OUTPUT_SIZE[1])
                    new_anno.append([xmin, ymin, xmax, ymax])
                    gt_class.append(CLASSES[bbox[0]])
            else:  # bottom-right
                img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y),interpolation=cv2.INTER_AREA)
                output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
                for bbox in img_annos:
                    xmin = scale_x + bbox[1] * (1 - scale_x)
                    ymin = scale_y + bbox[2] * (1 - scale_y)
                    xmax = scale_x + bbox[3] * (1 - scale_x)
                    ymax = scale_y + bbox[4] * (1 - scale_y)
                    xmin, ymin, xmax, ymax = int(xmin * OUTPUT_SIZE[0]), int(ymin * OUTPUT_SIZE[1]), int(
                        xmax * OUTPUT_SIZE[0]), int(ymax * OUTPUT_SIZE[1])
                    new_anno.append([xmin, ymin, xmax, ymax])
                    gt_class.append(CLASSES[bbox[0]])

        return output_img, new_anno, gt_class

    def get_dataset(self,imgsetpath):
        # class_id = category_name.index('person')

        img_paths = []
        annos = []

        # 读取指定txt文件的文件名
        xml_path = []


        for i in self.old_final_ids:
            file = i
            xml = self._annopath % "{0}".format(file)
            xml_path.append(xml)

        # print(glob.glob(os.path.join(anno_dir, '*.xml')))
        # for anno_file in glob.glob(os.path.join(anno_dir, '*.txt')):
        # for anno_file in glob.glob(os.path.join(anno_dir, '*.xml')):
        for i in range(len(xml_path)):
            anno_file = xml_path[i]
            anno_id = anno_file.split('/')[-1].split('.')[0]
            # print(anno_file)
            # anno_id = anno_file.split('\\')[-1].split('x')[0]
            # print('anno_id:' + str(anno_id))
            # set_trace()

            # with open(anno_file, 'r') as f:
            #     num_of_objs = int(f.readline())

            # set_trace()
            # print(img_dir)
            # img_path = os.path.join(img_dir, f'{anno_id}jpg')
            img_path = self._imgpath % "{0}".format(anno_id)
            # print(img_path)
            # print(img_path)
            img = cv2.imread(img_path)
            # set_trace()
            img_height, img_width, _ = img.shape
            # print(img.shape)
            del img

            boxes = []
            bnd_box = self.parseXmlFiles(anno_file)
            # print(bnd_box)
            for bnd_id, box in enumerate(bnd_box):
                # set_trace()

                categories_id = box[0]

                xmin = max(int(box[1]), 0) / img_width

                ymin = max(int(box[2]), 0) / img_height

                xmax = min(int(box[3]), img_width) / img_width

                ymax = min(int(box[4]), img_height) / img_height

                boxes.append([categories_id, xmin, ymin, xmax, ymax])
                # print(boxes)

                if not boxes:
                    continue

            img_paths.append(img_path)
            annos.append(boxes)
        # print("annos:所有对原图缩放后的坐标：", annos)
        # print(img_paths)
        return img_paths, annos

    def parseXmlFiles(self,anno_dir):
        tree = etree.parse(anno_dir)
        root = tree.getroot()
        objectes = root.findall('.//object')
        bnd_box = []
        for object in objectes:
            name = object.find("name").text

            bndbox = object.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            xmax = float(bndbox.find("xmax").text)
            ymin = float(bndbox.find("ymin").text)
            ymax = float(bndbox.find("ymax").text)

            # bnd_box.append([name, xmin, xmax, ymin, ymax])
            bnd_box.append([name, xmin, ymin, xmax, ymax])
            # print(len(bnd_box),bnd_box)
        return bnd_box

def main():
    data_dir = "/home/DATA/VOC2007"
    split = "test"  # train, val, test
    use_difficult = False
    transforms = None
    dataset = PascalVOCDataset(data_dir, split, use_difficult, transforms)


if __name__ == '__main__':
    main()
