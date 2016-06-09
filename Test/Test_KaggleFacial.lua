require 'torch'

-- test KaggleFacial dataset
global_args = {cuda = true}

local ds = require 'Dataset.Dataset_KaggleFacial'
print(ds.train_data:size())
print(ds.test_data:size())

-- create model
local model = require('Model.Model_Facial')
model.args.img_size = ds.train_data:size(3)
model.args.n_channels = ds.train_data:size(2)

model:create_model()
model:create_criterion()

print(model.model)

-- forward pass
local batch_data = ds.train_data[{ {5000,5050},{},{},{} }]
batch_data = batch_data:cuda()

local out = model.model:forward(batch_data)
print(out)