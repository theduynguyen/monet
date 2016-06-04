require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'

local Model_Base = torch.class('Model_Base')

function Model_Base:__init(args)
  self.args = args
  
  self.cuda = global_args.cuda
  self.model = nil
  self.criterion = nil
end

function Model_Base:create_model()
  error('Abstract class')
end

function Model_Base:create_criterion()
  error('Abstract class')
end

-- Convenience functions 
----------------------------------------------------------------------------
local padding_size = 2
local pooling_size = 2

-- Add convolution, BatchNorm and MaxPooling module
function add_conv(n_in,n_out,filter_size)
  local conv = nn.Sequential()
  conv:add(nn.SpatialConvolution(n_in, n_out, filter_size, filter_size))
  conv:add(nn.SpatialBatchNormalization(n_out))
  conv:add(nn.ReLU())               
  conv:add(nn.SpatialMaxPooling(pooling_size,pooling_size,pooling_size,pooling_size))
  
  return conv
end

-- Computes the output size of stacked conv layers
function get_n_out(conv,input_size)
  local n_channels = conv:get(1):get(1).nInputPlane
  local output = torch.Tensor(1,n_channels, input_size, input_size)
  output = conv:forward(output)
  
  return output:nElement(), output:size(3)
end

