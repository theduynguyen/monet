require 'Dataset.Dataset_Base'

require 'torch'
require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')
local Dataset_I40 = torch.class('Dataset_I40','Dataset_Base')

function Dataset_I40:__init(args)
  Dataset_Base:__init(args)
  
  self.hdf5_path = args
end

function Dataset_I40:load_ds()
  local file_found = io.open(self.hdf5_path,'r')
  
  if file_found == nil then
    print(self.hdf5_path)
    print('I40 Dataset not found')
    return
  end
  
  --load dataset
  local myFile = hdf5.open(self.hdf5_path, 'r')
  self.train_data = myFile:read('/train_data'):all()
  self.test_data = myFile:read('/test_data'):all()
  self.train_labels = myFile:read('/train_labels'):all()
  self.test_labels = myFile:read('/test_labels'):all()
  self.train_bbox = myFile:read('/train_bbox'):all()
  self.test_bbox = myFile:read('/test_bbox'):all()
  myFile:close()
  
  --compute sizes
  self.train_size = self.train_labels:size(1)
  self.test_size = self.test_labels:size(1)
end

local ds = Dataset_I40.new('/home/tdnguyen/Projects/I40_ObjDetect/I40_data_new.h5')
ds:load()
return ds