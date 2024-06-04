import os
import random
import argparse


# parser = argparse.ArgumentParser(description="sample_incremental dataset generator")
#
# parser.add_argument(
#     "dataset",
#     type=str,
#     default="defect_sampler_incremental"
# )
#
# args = parser.parse_args()

# if args.dataset == "defect_sampler_incremental":
#     data_dir = "/home/cgz/DATASET/FPC_source/VOCdevkit/VOC2007",
#     imgsetpath = os.path.join(data_dir, "ImageSets", "H_Sampler_incremental", "%s.txt")
#     image_set = "train"

# data_path = "/home/cgz/DATASET/FPC_source/VOCdevkit/VOC2007"
# imgsetpath = os.path.join(data_path, "ImageSets", "H_Sampler_incremental", "%s.txt")
# image_set = "train"
#
# with open(imgsetpath % image_set) as f:
#     ids = f.readlines()
#     print("ok")
#     # list2 = random.sample(list1, int(0.7 * len(list1)))


txt_all = "/home/cgz/DATASET/FPC_source/VOCdevkit/VOC2007/ImageSets/H_Sampler_incremental/train.txt"
txt_70 = "/home/cgz/DATASET/FPC_source/VOCdevkit/VOC2007/ImageSets/H_Sampler_incremental/train70.txt"


# list_all = []
# with open(txt_all) as f:
#     ids_all = f.readlines()
#     f.close()
# for i_all in ids_all:
#     list_all.append(i_all.split('\n')[0])


list_70 = []
with open(txt_70) as f:
    ids70 = f.readlines()
    f.close()
for idx_70 in ids70:
    list_70.append(idx_70.split('\n')[0])


# diff_list = list(set(list_all) - set(list_70))


txt_30 = "/home/cgz/DATASET/FPC_source/VOCdevkit/VOC2007/ImageSets/H_Sampler_incremental/train30.txt"
list_30 = []
with open(txt_30) as f:
    ids30 = f.readlines()
    f.close()
for idx_30 in ids30:
    list_30.append(idx_30.split('\n')[0])

set_same = set(list_30) & set(list_70)
list = list(set_same)

# with open(txt_30,'w') as f:
#     for index in diff_list:
#         f.write(index)
#         f.write("\n")
#     f.close()



print("ok")


    # list2 = random.sample(list1, int(0.7 * len(list1)))













