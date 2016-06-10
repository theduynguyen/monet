require 'Dataset.Dataset_Base'

require 'torch'
local mnist = require 'mnist'

local Dataset_MNIST = torch.class('Dataset_MNIST','Dataset_Base')

function Dataset_MNIST:__init(args)
  Dataset_Base:__init(args)
end

function Dataset_MNIST:load_ds()
  local trainset = mnist.traindataset()
  local testset = mnist.testdataset()
  
  
  self.train_data = torch.Tensor(trainset.data:size(1),1,trainset.data:size(2),trainset.data:size(3))
  for i = 1,trainset.data:size(1) do
    self.train_data[i][1] = trainset.data[i]
  end
    
  self.train_labels = trainset.label
  self.train_size = trainset.size
  
  self.test_data = torch.Tensor(testset.data:size(1),1,testset.data:size(2),testset.data:size(3))
  for i = 1,testset.data:size(1) do
    self.test_data[i][1] = testset.data[i]
  end
  
  self.test_labels = testset.label
  self.test_size = testset.size
end

local ds = Dataset_MNIST.new()
ds:load()
return ds