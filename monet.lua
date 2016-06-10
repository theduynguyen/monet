require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

-- command line params
cmd = torch.CmdLine()
cmd:text()
cmd:text('monet - MOdular NEural Trainer')

-- module params
cmd:option('--use_hypero', false, 'Use hypero logger')

-- experiment params
cmd:option('--batch_size', 50, 'batch size')
cmd:option('--epochs', 1, 'number of epochs')
cmd:option('--n_exp', 1, 'number of experiments')
cmd:option('--cuda', true, 'Use GPU')

cmd:option('--dataset', 'Dataset_I40', 'Dataset class')
cmd:option('--model', 'Model_CNN', 'Model')

-- architectural params
cmd:option('--dropout_p', '', 'dropout value')
cmd:option('--conv', '30,50,30', 'number of outputs for every conv layer')
cmd:option('--fc', '2000,1000', 'number of outputs for every fc layer')
cmd:option('--norm_glob', false, 'global contrast normalisation')
cmd:option('--norm_loc', false, 'local contrast normalisation')

-- optimizer params
cmd:option('--opt', 'Adam', 'optimizer : Adam | SGD ')
cmd:option('--learning_rate', 0.01, 'learning rate at t=0')
cmd:option('--lr_decay', 0.01, 'learning rate decay')
cmd:option('--w_decay', 0.01, 'weight decay')
cmd:option('--momentum', 0.9, 'SGD momentum')

global_args = cmd:parse(arg or {})
print 'global_args'
for k,v in pairs(global_args) do
    print(k,v)
end

-- create dataset
local ds = require('Dataset.'..global_args.dataset)

-- create model
local model = require('Model.'..global_args.model)
model.args.img_size = ds.train_data:size(3)
model.args.n_channels = ds.train_data:size(2)

-- only for classification: class number
model.args.n_classes = torch.max(ds.train_labels) + 1
-- how many outputs 
model.args.n_outs = ds.n_outs

model:create_model()
model:create_criterion()

-- create trainer
require 'Trainer.Sampler_Shuffle'
local sample = Sampler_Shuffle({batch_size = global_args.batch_size},ds)

require 'Model.ModelUpdater'
local mu = ModelUpdater

require 'Trainer.StoppingCrit'
local scrit = {}
table.insert(scrit,StoppingCrit_MaxIter({max_epoch = global_args.epochs}))

local components = {
  sampler = sample,
  modelUpdater = mu,
  model = model,
  stoppingCrit = scrit
}

require 'Trainer.Trainer'
local train = Trainer({},components)

-- register views
require 'View.View_Progress'
local v_progress = View_Progress({})
train:register('Batch',v_progress)

require 'View.View_Accuracy'
local v_acc = View_Accuracy({})
train:register('Epoch',v_acc)

if global_args.use_hypero == true then
  require 'View.View_Hypero'
  local v_hyp = View_Hypero({})
  train:register('Start',v_hyp)
  train:register('Stop',v_hyp)
  train:register('Epoch',v_hyp)
end

-- train
print('\n\nTrain! \n')
train:train()

-- save end model