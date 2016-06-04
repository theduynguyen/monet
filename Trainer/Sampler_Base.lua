require 'torch'

local Sampler_Base = torch.class('Sampler_Base')

function Sampler_Base:__init(args,dataset)
  self.args = args
  
  self.cuda = global_args.cuda
  self.dataset = dataset
end

function Sampler_Base:get_train_batch()
  error('Abstract class')
end

function Sampler_Base:get_test_batch()
  error('Abstract class')
end

function Sampler_Base:is_full_epoch()
  error('Abstract class')
end
