-- test MNIST dataset


local ds_mnist = require 'Dataset.Dataset_MNIST'

print(ds_mnist.test_data:size())

--[[
-- test I40 dataset
require 'Dataset.Dataset_I40'

local ds_i40 = Dataset_I40('/home/tdnguyen/Projects/I40_ObjDetect/I40_data_new.h5')
ds_i40:load()

-- test model creation
require 'Model.Model_CNN'

model_args = {}
model_args.n_channels = 3
model_args.img_size = 22
model_args.n_outs = 2
model_args.conv_layers = {100,50}
model_args.fc_layers = {1000,1000}
model_args.cuda = true

local m = Model_CNN(model_args)
m:create_model()
m:create_criterion()

-- test Preprocessor
require 'Trainer.Preprocessor_Normalize'
local preprocess_args = {}

local norm = Preprocessor_Normalize(preprocess_args)
local ds_norm, mean, std = norm:process(ds_i40.test_data)

-- test sampler
require 'Trainer.Sampler_Shuffle'

sample = Sampler_Shuffle({batch_size = 50, cuda = true},ds_i40)
data1, labels1 = sample:get_train_batch()


-- test train_epoch
require 'Model.ModelUpdater'
mu = ModelUpdater({})

require 'Trainer.StoppingCrit'
scrit = {}
table.insert(scrit,StoppingCrit_MaxIter({}))

components = {
  sampler = sample,
  modelUpdater = mu,
  model = m,
  stoppingCrit = scrit
}

require 'Trainer.Trainer'
train = Trainer({},components)

-- add Views
require 'View.View_Progress'
v_progress = View_Progress({})
train:register('Batch',v_progress)

require 'View.View_Accuracy'
v_acc = View_Accuracy({})
train:register('Epoch',v_acc)

train:train()
]]