require 'Model.Model_Base'

local Model_CNN = torch.class('Model_CNN','Model_Base')

function Model_CNN:__init(args)
  Model_Base:__init(args)
  
  self.args.conv_layers = {10,30}
  self.args.filter_sizes = {5,5}
  self.args.fc_layers = {120,60}
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
  
  -- save to make it easier for classification / regression layers
  self.fc_out = n_out
  
  --[[ assemble net ]]--
  local net = nn.Sequential()
  net:add(conv)
  net:add(fc)
          
  
  if global_args.cuda then
    cudnn.convert(net, cudnn)
    net:cuda()
  end
  
  self.model = net
end  

function Model_CNN:create_criterion()
  error('Abstract class')
end