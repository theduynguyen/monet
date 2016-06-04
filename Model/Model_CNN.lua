require 'Model.Model_Base'

local Model_CNN = torch.class('Model_CNN','Model_Base')

function Model_CNN:__init(args)
  Model_Base:__init(args)
end

function Model_CNN:create_model()
  
  --[[ Conv layers]]-- 
  local conv = nn.Sequential()
  local n_in = self.args.n_channels
  local n_out = 0
  
  for i=1,table.getn(self.args.conv_layers) do
    local n_out = self.args.conv_layers[i]
    conv:add(add_conv(n_in,n_out,self.args.filter_sizes[i]))
    n_in = n_out
  end
  
  --[[ FC layers]]-- 
  local fc = nn.Sequential()
  n_in = get_n_out(conv,self.args.img_size)
  fc:add(nn.Reshape(n_in))              
  n_out = 0
  
  for i=1,table.getn(self.args.fc_layers) do
    n_out = self.args.fc_layers[i]
    
    fc:add(nn.Linear(n_in, n_out))
    fc:add(nn.BatchNormalization(n_out))
    fc:add(nn.ReLU())
    n_in = n_out
  end
  
  local double_head = nn.ConcatTable()
  
  --[[ classifier ]]--
  local clf = nn.Sequential()
  clf:add(nn.Linear(n_out, self.args.n_outs))           
  clf:add(nn.LogSoftMax())
  --double_head:add(clf)
  
  --[[ bbox regressor ]]--
  --local reg = nn.Linear(n_out, self.args.n_outs) 
  --double_head:add(reg)
  
  --[[ assemble net ]]--
  local net = nn.Sequential()
  net:add(conv)
  net:add(fc)
  net:add(clf)
          
  
  if global_args.cuda then
    cudnn.convert(net, cudnn)
    net:cuda()
  end
  
  self.model = net
end  

function Model_CNN:create_criterion()
  --local crit = nn.ParallelCriterion()
  
  --[[ classification ]]--
  --local crit_clf = nn.CrossEntropyCriterion()
  --crit:add(crit_clf)  
  
  --[[ bbox regression ]]--
  --local crit_reg = nn.MSECriterion()
  --crit:add(crit_reg)
  
  local crit = nn.CrossEntropyCriterion()
  
  if global_args.cuda then
    crit:cuda()
  end
    
  self.criterion = crit
end


model_args = {}
model_args.n_channels = 3
model_args.img_size = 22
model_args.n_outs = 2

model_args.conv_layers = {10,30}
model_args.filter_sizes = {5,5}
model_args.fc_layers = {120,60}


local m = Model_CNN.new(model_args)
return m