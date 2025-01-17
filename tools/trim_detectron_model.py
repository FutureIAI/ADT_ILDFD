import os
import torch
import argparse
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--name",
    default="defect_6_5",
    help="Exp name",
    type=str,
)
parser.add_argument(
    "--instance",
    default=False,
    action='store_true',
)
args = parser.parse_args()


base_dir = 'mask_out' if args.instance else "output"

name = args.name
args.pretrained_path = "/media/cgz/新加卷/XWJ/FPC11/source_model/0-6/model_0007500.pth"
args.save_path = "/media/cgz/新加卷/XWJ/FPC11/source_model/0-6/model_7500.pth"
PRETRAINED_PATH = os.path.expanduser(args.pretrained_path)
print('pretrained model path: {}'.format(PRETRAINED_PATH))

# remove optimizer and iteration information, only remain model parameter and structure information
pretrained_weights = torch.load(PRETRAINED_PATH)['model']
# print('pretrained weights: {0}'.format(pretrained_weights))

new_dict = {k: v for k, v in pretrained_weights.items()}

# print('new dict: {0}'.format(new_dict))

torch.save(new_dict, args.save_path)
print('saved to {}.'.format(args.save_path))
