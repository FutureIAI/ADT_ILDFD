import random

import cv2
import torch
import os
import glob
import numpy as np
from PIL import Image
from lxml import etree
# from ipdb import set_trace
from createXML import GEN_Annotations,createXml

OUTPUT_SIZE = (768*2, 768*2)  # Height, Width
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.

# voc格式的数据集，anno_dir是标注xml文件，img_dir是对应jpg图片
ANNO_DIR = r'C:\Users\CGZ\Desktop\FPC_source\VOCdevkit\VOC2007\Annotations'
IMG_DIR = r'C:\Users\CGZ\Desktop\FPC_source\VOCdevkit\VOC2007\JPEGImages'
# category_name = ['background', 'person']

CLASSES = {"__background__ ":0, "Z1":1, "Z2":2, "Z3":3,"Z4":4,"C3":5 ,"C4":6 ,"F2":7 ,"G1":8 ,"J3":9 }

img_id = 0
save_img = r'C:\Users\CGZ\Desktop\detection\JPEGImages'
save_img_bbox = r'C:\Users\CGZ\Desktop\detection\JPEGImages_bbox'
save_xml = r'C:\Users\CGZ\Desktop\detection\Annotations'
def mosaic(img_id,old_final_ids,annopath,imgpath,imgsetpath):

    img_paths, annos = get_dataset(old_final_ids,annopath, imgpath,imgsetpath)
    # print('长度:' + str(len(img_paths)) + ';' + str(len(annos)))
    # print(len(img_paths))
    # print(len(annos))
    # print(type(img_paths))
    # print(type(annos))
    # print(img_paths)
    # print(annos)
    # set_trace()

    idxs = random.sample(range(len(annos)), 3)  # 从annos列表长度中随机取3个数
    print(idxs)
    idxs.append(old_final_ids.index(img_id))
    print(idxs)
    random.shuffle(idxs)
    # print(idxs)
    # set_trace()

    new_image, new_annos, gt_class = update_image_and_anno(img_paths, annos,
                                                 idxs,
                                                 OUTPUT_SIZE, SCALE_RANGE,
                                                 filter_scale=FILTER_TINY_SCALE)

    difficult =  torch.zeros(len(new_annos))
    im_info = tuple(map(int, (new_image.shape[1], new_image.shape[0])))
    return new_image,{
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


def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    gt_class = []
    for i, idx in enumerate(idxs):
        # set_trace()
        path = all_img_list[idx]
        img_annos = all_annos[idx]

        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * scale_x
                ymax = bbox[4] * scale_y
                xmin,ymin,xmax,ymax = int(xmin*OUTPUT_SIZE[0]),int(ymin*OUTPUT_SIZE[1]),int(xmax*OUTPUT_SIZE[0]),int(ymax*OUTPUT_SIZE[1])
                new_anno.append([xmin, ymin, xmax, ymax])
                gt_class.append(CLASSES[bbox[0]])
        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
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
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
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
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
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

    return output_img, new_anno,  gt_class


def get_dataset(old_final_ids,anno_dir, img_dir,imgsetpath):
    # class_id = category_name.index('person')

    img_paths = []
    annos = []

    # 读取指定txt文件的文件名
    xml_path = []

    filenames = old_final_ids
    for i in filenames:
        file = i
        xml = anno_dir % "{0}".format(file)
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
        img_path = img_dir % "{0}".format(anno_id)
        # print(img_path)
        # print(img_path)
        img = cv2.imread(img_path)
        # set_trace()
        img_height, img_width, _ = img.shape
        # print(img.shape)
        del img

        boxes = []
        bnd_box = parseXmlFiles(anno_file)
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


def parseXmlFiles(anno_dir):
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


if __name__ == '__main__':

    # main()

    for i in range(2000):
        main()
        print('第' + str(img_id) + '张图片已经生成!')
        img_id = img_id + 1

    pass



