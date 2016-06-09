require 'Model.Model_CNN'

local Model_Facial = torch.class('Model_Facial','Model_CNN')

function Model_Facial:__init(args)
  Model_CNN:__init(args)
end

function Model_Facial:create_model()
  --create CNN and FC layers
  Model_CNN:create_model()
  
  --[[ regressor ]]--
  self.model:add(nn.Linear(self.fc_out, 1))
          
  if global_args.cuda then
    cudnn.convert(self.model, cudnn)
    self.model:cuda()
  end
end  

function Model_Facial:create_criterion()
  local crit = nn.MSECriterion()
  
  if global_args.cuda then
    crit:cuda()
  end
    
  self.criterion = crit
end

local model_args = {}
model_args.n_channels = 1
model_args.img_size = 96
model_args.n_outs = 1

local m = Model_Facial.new(model_args)
return m