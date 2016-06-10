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
  -- has to be implemented by child class
  self:load_ds()
  
  -- compute properties and checks from dataset
  self:compute_properties()
end

function Dataset_Base:load_ds()
  error('Abstract class')
end

function Dataset_Base:compute_properties()
  -- determine number of label outputs
  if self.train_labels:dim() == 1 then
    self.n_outs = 1
  else
    self.n_outs = self.train_labels:size(2)
  end
end