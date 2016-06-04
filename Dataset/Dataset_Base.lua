require 'torch'

--Base class for datasets: defines train/test data and labels, can load from various sources
local Dataset_Base = torch.class('Dataset_Base')

function Dataset_Base:__init(args)
  self.train_data = torch.FloatTensor()
  self.test_data = torch.FloatTensor()
  
  self.train_labels = 0
  self.test_labels = 1
  
  self.train_size = 0
  self.test_size = 0
end

--virtual method to fill the variables defined in constructor
function Dataset_Base:load()
  error('Abstract class')
end