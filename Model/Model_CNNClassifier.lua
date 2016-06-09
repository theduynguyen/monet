require 'Model.Model_CNN'

local Model_CNNClassifier = torch.class('Model_CNNClassifier','Model_CNN')

function Model_CNNClassifier:__init(args)
  Model_CNN:__init(args)
end

function Model_CNNClassifier:create_model()
  --create CNN and FC layers
  Model_CNN:create_model()
  
  --[[ classifier ]]--
  local clf = nn.Sequential()
  clf:add(nn.Linear(self.fc_out, self.args.n_outs))           
  clf:add(nn.LogSoftMax())           
  
  --[[ assemble net ]]--
  self.model:add(clf)
          
  if global_args.cuda then
    cudnn.convert(self.model, cudnn)
    self.model:cuda()
  end
end  

function Model_CNNClassifier:create_criterion()
  local crit = nn.CrossEntropyCriterion()
  
  if global_args.cuda then
    crit:cuda()
  end
    
  self.criterion = crit
end

local model_args = {}
model_args.n_channels = 1
model_args.img_size = 96
model_args.n_outs = 1

local m = Model_CNNClassifier.new(model_args)
return m