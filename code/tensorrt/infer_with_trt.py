import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torchvision.transforms
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import cv2
import torch
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

parser = argparse.ArgumentParser(
  description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--models', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')


def load_engine(trt_runtime, engine_path):
  with open(engine_path, 'rb') as f:
    engine_data = f.read()
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  return engine

class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()

def allocate_buffers_for_encoder(engine):
  """Allocates host and device buffer for TRT engine inference.
  This function is similair to the one in common.py, but
  converts network outputs (which are np.float32) appropriately
  before writing them to Python buffer. This is needed, since
  TensorRT plugins doesn't support output type description, and
  in our particular case, we use NMS plugin as network output.
  Args:
      engine (trt.ICudaEngine): TensorRT engine
  Returns:
      inputs [HostDeviceMem]: engine input memory
      outputs [HostDeviceMem]: engine output memory
      bindings [int]: buffer to device bindings
      stream (cuda.Stream): cuda stream for engine inference synchronization
  """
  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  binding_to_type = {}
  binding_to_type['input'] = np.float32

  binding_to_type['output'] = np.float32
  binding_to_type['input.43'] = np.float32
  binding_to_type['input.79'] = np.float32
  binding_to_type['input.115'] = np.float32
  binding_to_type['192'] = np.float32

  # Current NMS implementation in TRT only supports DataType.FLOAT but
  # it may change in the future, which could brake this sample here
  # when using lower precision [e.g. NMS output would not be np.float32
  # anymore, even though this is assumed in binding_to_type]


  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume * engine.max_batch_size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.

class TRTInference(object):
  """Manages TensorRT objects for model inference."""

  def __init__(self, trt_engine_path=None, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1):
    """Initializes TensorRT objects needed for model inference.
    Args:
        trt_engine_path (str): path where TensorRT engine should be stored
        uff_model_path (str): path of .uff model
        trt_engine_datatype (trt.DataType):
            requested precision of TensorRT engine used for inference
        batch_size (int): batch size for which engine
            should be optimized for
    """

    # We first load all custom plugins shipped with TensorRT,
    # some of them will be needed during inference
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    # Initialize runtime needed for loading TensorRT engine from file
    self.trt_runtime = trt.Runtime(TRT_LOGGER)
    # TRT engine placeholder
    self.trt_engine = None

  def infer(self, image_path):
    """Infers model on given image.
    Args:
        image_path (str): image to run object detection model on
    """

    # Load image into CPU
    # this part is for the encoder inference
    print(f'[trace] TensorRT inference for input image{image_path}')
    # Copy it into appropriate place into memory
    # (self.inputs was returned earlier by allocate_buffers())
    # When infering on single image, we measure inference
    # time to output it to the user
    # infer for the first engine
    batch_size = 1

    trt_engine_path = './models/encoder-simp.engine'
    if (os.path.exists(trt_engine_path) == False):
      print(f'[trace] engine file {trt_engine_path} does not exist, exit')
      exit(-1)

    print("[trace] Loading cached TensorRT engine from {}".format(trt_engine_path))
    self.trt_engine = load_engine(self.trt_runtime, trt_engine_path)
    # This allocates memory for network inputs/outputs on both CPU and GPU
    print("[trace] TensorRT engine loaded")
    print("[trace] allocating buffers for TensorRT engine")
    self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers_for_encoder(self.trt_engine)
    print("[trace] allocating buffers done")
    # Execution context is needed for inference

    print("[trace] TensorRT engine: creating execution context")
    self.context = self.trt_engine.create_execution_context()

    img = self._load_img(image_path)
    np.copyto(self.inputs[0].host, img.ravel())
    [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
    # Run inference.
    self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
    # Transfer predictions back from the GPU.
    # inspect the output object here;


    '''
    print('[trace] cuda.memcpy_dtoh_async')
    [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
    # Synchronize the stream
    print('[trace] stream.synchronize()')
    self.stream.synchronize()
    # Return only the host outputs.
    # return [out.host for out in outputs]
    
    # Output inference time
    print("TensorRT inference time: {} ms".format(
      int(round((time.time() - inference_start_time) * 1000))))
    print("[trace] TensorRT inference time: {} ns".format(
      int(round((inference_end_time_ns - inference_start_time_ns)))))
    '''

    # =======================================================================================
    # this part is for the decoder inference
    trt_engine_path = './models/depth_decoder-simp.engine'
    if (os.path.exists(trt_engine_path) == False):
      print(f'[trace] engine file {trt_engine_path} does not exist, exit')
      exit(-1)

    self.trt_engine = load_engine(self.trt_runtime, trt_engine_path)
    # This allocates memory for network inputs/outputs on both CPU and GPU
    print("[trace] decoder engine loaded")
    print("[trace] allocating buffers for TensorRT engine")

    self.bindings = []
    self.inputs = self.outputs
    self.outputs = []

    binding_to_type = {}
    binding_to_type['input0'] = np.float32
    binding_to_type['input1'] = np.float32
    binding_to_type['input2'] = np.float32
    binding_to_type['input3'] = np.float32
    binding_to_type['input4'] = np.float32

    binding_to_type['disp0'] = np.float32
    binding_to_type['disp1'] = np.float32
    binding_to_type['disp2'] = np.float32
    binding_to_type['disp3'] = np.float32

    for _input in self.inputs:
      self.bindings.append(int(_input.device))

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    engine = self.trt_engine
    for binding in engine:
      _binding_shape = engine.get_binding_shape(binding)
      _volume = trt.volume(_binding_shape)
      size = _volume * engine.max_batch_size

      dtype = binding_to_type[str(binding)]
      # Allocate host and device buffers
      # Append to the appropriate list.
      if engine.binding_is_input(binding):
        #inputs.append(HostDeviceMem(host_mem, device_mem))
        pass
      else:
        print(f'[trace] current output binding: {str(binding)}')
        print(f'[trace] current output binding size: {size}')
        print(f'[trace] current output dtype: {dtype}')
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        self.bindings.append(int(device_mem))
        print(f'[trace] current output device_mem: {device_mem}')
        # Append the device buffer to device bindings.
        self.outputs.append(HostDeviceMem(host_mem, device_mem))

    print("[trace] allocating buffers done")
    # Execution context is needed for inference
    print("[trace] TensorRT engine: creating execution context")
    #self.context = self.trt_engine.create_execution_context()
    self.context = self.trt_engine.create_execution_context()
    self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
    self.stream.synchronize()
    [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
    self.stream.synchronize()

    retObj = self.outputs
    disp3 = retObj[0]
    disp2 = retObj[1]
    disp1 = retObj[2]
    disp0 = retObj[3]
    original_height = 192
    original_width = 640
    disp = torch.from_numpy(disp0.host)
    disp = torch.reshape(disp, (1,1,192,640))

    disp_resized = torch.nn.functional.interpolate(
      disp, (original_height, original_width), mode="bilinear", align_corners=False)
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    im.show()

    return
    # And return results
    #return detection_out, keepCount_out

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    new_shape = (im_height, im_width, 1)
    return np.array(image).reshape(new_shape).astype(np.uint8)

  def _load_img(self, image_path):
    im = Image.open(image_path)
    im = im.resize((640, 192))
    #im.show()
    trans_to_tensor = torchvision.transforms.ToTensor()
    _tensor = trans_to_tensor(im)
    return _tensor
    '''
    image_height = 112
    image_width = 112
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    dim = (image_width, image_height)  # resize im
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized_image_tensor = torch.from_numpy(resized_image)
    resized_image_tensor = resized_image_tensor.float()
    resized_image_tensor = resized_image_tensor.view(1, -1, image_width, image_height)
    print(f'[trace] resized_image_tensor.size(): {resized_image_tensor.size()}')
    #using arcface-resnet100.engine as the inference engine

    return resized_image_tensor

    '''


def infer_with_torch_network():

  print(f'[trace] @infer_with_torch_network')
  _encoder_weight_file = './models/mono+stereo_640x192/encoder.pth'
  if not os.path.exists(_encoder_weight_file):
    print(f'[trace] target encode weight file {_encoder_weight_file} does not exist')
    exit(-1)

  print(f'[trace] load network')
  encoder = networks.ResnetEncoder(18, False)
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  print(f'[trace] load encoder weight')
  loaded_dict_enc = torch.load(_encoder_weight_file, map_location=device)
  feed_height = loaded_dict_enc['height']
  feed_width = loaded_dict_enc['width']
  filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
  encoder.load_state_dict(filtered_dict_enc)
  encoder.to(device)
  encoder.eval()

  print(f'[trace] load image')
  _image_file_path = './assets/test_image.jpg'
  if not os.path.exists(_image_file_path):
    print(f'[trace] target image file {_image_file_path} does not exist')
    exit(-1)

  print(f'[trace] image to tensor')
  input_image = pil.open(_image_file_path).convert('RGB')
  original_width, original_height = input_image.size
  input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
  input_image = transforms.ToTensor()(input_image).unsqueeze(0)

  # PREDICTION
  print(f'[trace] start to predict')
  input_image = input_image.to(device)
  features = encoder(input_image)

  _depth_decoder_path = './models/mono+stereo_640x192/depth.pth'
  if not os.path.exists(_depth_decoder_path):
    print(f'[trace] depth decoder weight file {_depth_decoder_path} does not exist')
    exit(-1)
  loaded_dict = torch.load(_depth_decoder_path, map_location=device)
  depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
  depth_decoder.load_state_dict(loaded_dict)
  depth_decoder.to(device)
  depth_decoder.eval()

  outputs = depth_decoder(*features)
  disp = outputs[0]
  disp_resized = torch.nn.functional.interpolate(
    disp, (original_height, original_width), mode="bilinear", align_corners=False)
  disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
  vmax = np.percentile(disp_resized_np, 95)
  normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
  mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
  colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
  im = pil.fromarray(colormapped_im)
  im.show()
  pass

def main():

  print("[trace] reach the main entry")

  '''
  infer_with_torch_network()
  return
  '''

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.cuda else "cpu")
  input_image_path = './assets/test_image.jpg'
  if (os.path.exists(input_image_path) == False):
    print(f'[trace] target file {input_image_path} does not exist, exit')
    exit(-1)

  print('[trace] initiating TensorRT object')
  trtObject = TRTInference()
  trtObject.infer(input_image_path)
  print(f'[trace] end of the main point')
  pass

if __name__ == "__main__":
  main()
  pass
