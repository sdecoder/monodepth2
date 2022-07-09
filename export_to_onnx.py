import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
  parser = argparse.ArgumentParser(
    description='Simple testing funtion for Monodepthv2 models.')

  parser.add_argument('--model_name', type=str,
                      help='name of a pretrained model to use',
                      choices=[
                        "mono_640x192",
                        "stereo_640x192",
                        "mono+stereo_640x192",
                        "mono_no_pt_640x192",
                        "stereo_no_pt_640x192",
                        "mono+stereo_no_pt_640x192",
                        "mono_1024x320",
                        "stereo_1024x320",
                        "mono+stereo_1024x320"])
  parser.add_argument("--no_cuda",
                      help='if set, disables CUDA',
                      action='store_true')
  parser.add_argument("--pred_metric_depth",
                      help='if set, predicts metric depth instead of disparity. (This only '
                           'makes sense for stereo-trained KITTI models).',
                      action='store_true')

  return parser.parse_args()


def test_simple(args):
  """Function to predict for a single image or folder of images
  """
  assert args.model_name is not None, \
    "You must specify the --model_name parameter; see README.md for an example"

  if torch.cuda.is_available() and not args.no_cuda:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  if args.pred_metric_depth and "stereo" not in args.model_name:
    print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
          "models. For mono-trained models, output depths will not in metric space.")

  download_model_if_doesnt_exist(args.model_name)
  model_path = os.path.join("models", args.model_name)
  print("-> Loading model from ", model_path)
  encoder_path = os.path.join(model_path, "encoder.pth")
  depth_decoder_path = os.path.join(model_path, "depth.pth")

  # LOADING PRETRAINED MODEL
  print("   Loading pretrained encoder")
  encoder = networks.ResnetEncoder(18, False)
  loaded_dict_enc = torch.load(encoder_path, map_location=device)
  # extract the height and width of image that this model was trained with
  feed_height = loaded_dict_enc['height']
  feed_width = loaded_dict_enc['width']
  print(f'[trace] feed_height: {feed_height}')
  print(f'[trace] feed_width: {feed_width}')
  filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
  encoder.load_state_dict(filtered_dict_enc)
  encoder.to(device)
  encoder.eval()

  print("[trace] Loading pretrained decoder")
  depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

  loaded_dict = torch.load(depth_decoder_path, map_location=device)
  depth_decoder.load_state_dict(loaded_dict)

  depth_decoder.to(device)
  depth_decoder.eval()

  # print(f'[trace] checking the encoder mode: {encoder}')

  print(f'[trace] the encoder model structure:\n{encoder}')
  print(f'[trace] checking the depth_decoder mode:\n{depth_decoder}')
  batch_size = 1
  channel = 3
  input = torch.randn(batch_size, channel, feed_height, feed_width, requires_grad=True)
  input = input.to(device)
  output_model_name = 'models/encoder.onnx'
  # Export the model
  '''
  torch.onnx.export(encoder,  # model being run
                    input,  # model input (or a tuple for multiple inputs)
                    output_model_name,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=14,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output'])
  print(f'[trace] the encoder model has been exported')

  '''

  input0 = torch.FloatTensor(1, 64, 96, 320).to(device)
  input1 = torch.FloatTensor(1, 64, 48, 160).to(device)
  input2 = torch.FloatTensor(1, 128, 24, 80).to(device)
  input3 = torch.FloatTensor(1, 256, 12, 40).to(device)
  input4 = torch.FloatTensor(1, 512, 6, 20).to(device)
  input_tuple = (input0, input1, input2, input3, input4)
  output_model_name = 'models/depth_decoder.onnx'

  '''
  example_output = depth_decoder(input0, input1, input2, input3, input4)
  print('[trace] done with test run for depth_decoder(input_tuple)')
  torch.onnx.export(depth_decoder,  # model being run
                    args=input_tuple,  # model input (or a tuple for multiple inputs)
                    f=output_model_name,)
  '''

  torch.onnx.export(depth_decoder,  # model being run
                    args=input_tuple,  # model input (or a tuple for multiple inputs)
                    f=output_model_name,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=12,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input0', 'input1', 'input2', 'input3', 'input4'],  # the model's input names
                    output_names=['disp0', 'disp1', 'disp2', 'disp3'])


  print(f'[trace] the depth-decoder model has been exported')
  pass


def main(args):
  print(f'[trace] working in the main function')
  test_simple(args)
  pass


if __name__ == '__main__':
  args = parse_args()
  main(args)
