# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import argparse
import sys
import time

from torch.autograd import Variable
from tqdm import tqdm

from bts_dataloader import *


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=150)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_output', type=bool, help='if set, save outputs', default=True)
parser.add_argument('--bts_size', type=int, help='initial num_filters in bts', default=512)
parser.add_argument('--output', type=str, help='path to the results', default='~')
# focal value doesn't really change anything
parser.add_argument('--focal', type=float, help='override the filenames.txt value for the focal', default=721)
parser.add_argument('--image_width', type=int, help='image width', default=512)
parser.add_argument('--image_height', type=int, help='image height', default=256)
parser.add_argument('--input_size', type=int,
                    help='input size, cropped and augmented square images input into the network', default=256)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

# focal = (image_width / 2) / tan(FOV / 2), in this dataset FOV = 0.887981
args.focal = int((args.image_width / 2) / np.tan(0.887981 / 2))

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args)

    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    start_time = time.time()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(torch.cuda.FloatTensor(args.focal))
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

            if args.mode == "test" and args.save_output:
                filename = lines[i].split()[0].split('/')[-1].split('.')[0]
                filename_rgb = "{}_{}_{}.png".format(filename, args.max_depth, args.focal)
                filename = "{}_{}_{}_{}.png".format(filename, args.max_depth, args.focal, args.model_name)
                filename_pred_png = os.path.join(args.output, filename)
                filename_rgb_png = os.path.join(args.output, filename_rgb)

                if args.dataset == 'nyu':
                    gt_path = os.path.join(args.data_path, lines[i].split()[1])
                    gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
                    gt[gt == 0] = np.amax(gt)

                pred_depth = pred_depths[i]
                pred_8x8 = pred_8x8s[i]
                pred_4x4 = pred_4x4s[i]
                pred_2x2 = pred_2x2s[i]
                pred_1x1 = pred_1x1s[i]

                if args.dataset == "nyu":
                    pred_depth_scaled = pred_depth * 1000.0
                if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
                    pred_depth_scaled = pred_depth * 256.0

                cv2.imwrite(filename_rgb_png, image.cpu().numpy().squeeze().transpose((1, 2, 0)) * 255,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    elapsed_time = time.time() - start_time
    print('Elapsed time: %s' % str(elapsed_time))
    print('Done.')
    print('Saving result pngs in', os.path.dirname(filename_rgb_png))

    return


test(args)
