import argparse

import onnx
import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum
import utility

def generate_trt_engine():
  import pandas as pd

  onnx_file_path = '../../models/depth_decoder-simp.onnx'
  print(f'[trace] convert onnx file {onnx_file_path} to TensorRT engine')
  if not os.path.exists(onnx_file_path):
    print(f'[trace] target file {onnx_file_path} not exist, exiting')
    exit(-1)

  print(f'[trace] exec@generate_trt_engine')

  [trainloader, valloader, vis_dataloader] = utility.prepare_train_data()
  _imgs0, _imgs1, _labels = iter(trainloader).next()
  cache_file = "calibration.cache"
  # 4 is the lenghth of fp32
  element_bytes = _imgs0.shape[1] * _imgs0.shape[2] * _imgs0.shape[3] * 4
  batch_size = 64

  calib = utility.Calibrator(trainloader, cache_file, element_bytes, batch_size=batch_size)
  # engine = build_engine_from_onnxmodel_int8(onnxmodel, calib)

  mode: utility.CalibratorMode = utility.CalibratorMode.INT8
  TRT_LOGGER = trt.Logger()
  EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

  with trt.Builder(TRT_LOGGER) as builder, \
      builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, \
      trt.OnnxParser(network, TRT_LOGGER) as parser, \
      trt.Runtime(TRT_LOGGER) as runtime:

    # Parse model file
    print("[trace] loading ONNX file from path {}...".format(onnx_file_path))
    with open(onnx_file_path, "rb") as model:
      print("[trace] beginning ONNX file parsing")
      if not parser.parse(model.read()):
        print("[error] failed to parse the ONNX file.")
        for error in range(parser.num_errors):
          print(parser.get_error(error))
        return None
      print("[trace] completed parsing of ONNX file")

    builder.max_batch_size = batch_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, utility.GiB(4))

    if mode == utility.CalibratorMode.INT8:
      config.set_flag(trt.BuilderFlag.INT8)
    elif mode == utility.CalibratorMode.FP16:
      config.set_flag(trt.BuilderFlag.FP16)
    elif mode == utility.CalibratorMode.TF32:
      config.set_flag(trt.BuilderFlag.TF32)
    elif mode == utility.CalibratorMode.FP32:
      # do nothing since this is the default branch
      # config.set_flag(trt.BuilderFlag.FP32)
      pass
    else:
      print(f'[trace] unknown calibrator mode: {mode.name}, exit')
      exit(-1)

    config.int8_calibrator = calib
    engine_file_path = f'../models/siamese_network.{mode.name}.engine'
    input_channel = 3
    input_image_width = 100
    input_image_height = input_image_width
    network.get_input(0).shape = [batch_size, input_channel, input_image_width, input_image_height]
    network.get_input(1).shape = [batch_size, input_channel, input_image_width, input_image_height]

    print(f'[trace] utility.build_engine_common_routine')
    return utility.build_engine_common_routine(network, builder, config, runtime, engine_file_path)
  pass


def _export2onnx_internal(model, onnx_file_name, input_shape):
  print(f'[trace] working in the func: create_onnx_file')
  if os.path.exists(onnx_file_name):
    print(f'[trace] {onnx_file_name} exist, return')
    return onnx_file_name

  print(f'[trace] start to export the torchvision resnet50')
  input_name = ['input0', 'input1']
  output_name = ['output']
  from torch.autograd import Variable

  '''
  import torch.nn as nn
  # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model.fc = nn.Linear(2048, 10, bias=True)
  # Input to the model
  # [trace] input shape: torch.Size([16, 3, 512, 512])  
  batch_size = 16
  channel = 3
  pic_dim = 512
  '''

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input0 = torch.randn(input_shape, requires_grad=True).to(device)
  input1 = torch.randn(input_shape, requires_grad=True).to(device)
  inputs = (input0, input1)
  '''
  output = model(input)
  print(f'[trace] model output: {output.size()}')
    dynamic_axes = {
    'input': {0: 'batch_size'}
  }
  '''

  # Export the model
  torch.onnx.export(model,  # model being run
                    inputs,  # model input (or a tuple for multiple inputs)
                    onnx_file_name,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=16,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=input_name,  # the model's input names
                    output_names=output_name,  # the model's output names
                    dynamic_axes={'input0': {0: 'batch_size'},  # variable length axes
                                  'input1': {0: 'batch_size'},  # variable length axes
                                  'output': {0: 'batch_size'}})

  print(f'[trace] done with onnx file exporting')
  # modify the network to adapat MNIST
  pass

def _main():
  print(f'[trace] working in the main function')
  generate_trt_engine()
  pass


if __name__ == '__main__':
  _main()
