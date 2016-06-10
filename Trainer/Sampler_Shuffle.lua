require 'torch'
require 'cutorch'
require 'Trainer.Sampler_Base'

local Sampler_Shuffle = torch.class('Sampler_Shuffle','Sampler_Base')

function Sampler_Shuffle:__init(args,dataset)
  Sampler_Base:__init(args)
  
  self.full_epoch = false
  self.dataset = dataset
  self.batch_size = args.batch_size or 50
  
  -- compute number of batches
  self.n_train_batches = math.floor(self.dataset.train_size / self.batch_size)
  self.n_test_batches = math.floor(self.dataset.test_size / self.batch_size)
  
  self.current_train_batch = 0
  self.current_test_batch = 0
  
  self.test_idx = torch.range(1,self.dataset.test_size)
  self.train_idx = torch.randperm(self.dataset.train_size)
  
  -- start shuffling dataset
  self:new_epoch()
end

function Sampler_Shuffle:get_train_batch()
  local batch = self:get_data_batch(self.current_train_batch,self.train_idx,
                                    self.dataset.train_data, self.dataset.train_labels, self.dataset.train_size)
  
  self.current_train_batch = self.current_train_batch + 1
  
  return batch
end

function Sampler_Shuffle:get_test_batch()
  local batch = self:get_data_batch(self.current_test_batch,self.test_idx,
                                    self.dataset.test_data,self.dataset.test_labels,self.dataset.test_size)
  
  self.current_test_batch = self.current_test_batch + 1
  
  return batch
end

function Sampler_Shuffle:get_data_batch(batch_id,idx,
                                        data,labels,dataset_size)
  -- compute indices
  local min_idx = batch_id * self.batch_size
  local max_idx = math.min(min_idx + self.batch_size, dataset_size)
  local batch_size = max_idx - min_idx
  
  -- allocate space
  local n_channels = data:size(2)
  local img_size = data:size(3)
  
  local batch_data = torch.Tensor(batch_size,n_channels,img_size,img_size)
  local batch_labels = torch.Tensor(batch_size,self.dataset.n_outs)
  
  -- copy dataset data to allocated space
  for i=1,batch_size do
    batch_data[i] = data[idx[min_idx+i]]
    batch_labels[i] = labels[idx[min_idx+i]]
  end
  
  --1 LUA's indexing
  batch_labels = batch_labels+1
  
  --upload to CUDA
  if self.cuda then
    batch_data = batch_data:cuda()
    batch_labels = batch_labels:cuda()
  end 
  
  batch = {}
  batch.inputs = batch_data
  batch.targets = batch_labels
    
  return batch
end

function Sampler_Shuffle:is_full_epoch()
  if self.current_train_batch >= self.n_train_batches then
    self:new_epoch()
    
    return true
  end
  
  if self.current_test_batch >= self.n_test_batches then
    self:new_epoch()
    
    return true
  end
  
  return false
end
function Sampler_Shuffle:new_epoch()
  self.current_train_batch = 0
  self.current_test_batch = 0
    
  self.train_idx = torch.randperm(self.dataset.train_size)
end

