require 'Dataset.Dataset_Base'

require 'torch'
require 'hdf5'

local Dataset_KaggleFacial = torch.class('Dataset_KaggleFacial','Dataset_Base')

function Dataset_KaggleFacial:__init(args)
  Dataset_Base:__init(args)
  
  self.hdf5_path = args
end

function Dataset_KaggleFacial:load()
  local file_found = io.open(self.hdf5_path,'r')
  
  if file_found == nil then
    print(self.hdf5_path)
    print('Kaggle Facial Dataset not found')
    return
  end
  
  --load dataset
  local myFile = hdf5.open(self.hdf5_path, 'r')
  self.train_data = myFile:read('/train_data'):all()
  self.test_data = myFile:read('/test_data'):all()
  self.train_labels = myFile:read('/train_labels'):all()
  self.test_labels = myFile:read('/test_labels'):all()
  myFile:close()
  
  --normalize dataset (current images are in char values)
  self.train_data = self.train_data:float():div(255.0)
  self.test_data = self.test_data:float():div(255.0)
  --self.train_labels = self.train_labels:float():div(96.0)
  --self.test_labels = self.test_labels:float():div(96.0)
  
  --compute sizes
  self.train_size = self.train_labels:size(1)
  self.test_size = self.test_labels:size(1)
end

local ds = Dataset_KaggleFacial.new('/home/tdnguyen/Projects/Kaggle_Facial/kaggle_facial.h5')
ds:load()
return ds