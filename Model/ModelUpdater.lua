require 'torch'
require 'optim'

local ModelUpdater = torch.class('ModelUpdater')

-- Template method - need to specify blocks
function ModelUpdater:__init(args)
  self.args = args
end

function ModelUpdater:step(model,batch)
  local opt_params = {
     learningRate = global_args.learning_rate,
     learningRateDecay = global_args.lr_decay,
     weightDecay = global_args.w_decay,
     momentum = global_args.momentum
  }
  
  x, dl_dx = model.model:getParameters()
  
  local current_loss = 0
  local count = 0

  -- evaluation
  local feval = function(x_new)
      -- reset data
      if x ~= x_new then x:copy(x_new) end
      dl_dx:zero()
      
      -- perform mini-batch gradient descent
      local outputs = model.model:forward(batch.inputs)
      
      local loss = model.criterion:forward(outputs, batch.targets)
      model.model:backward(batch.inputs, model.criterion:backward(model.model.output, batch.targets))
      
      return loss, dl_dx
  end
  
  if global_args.opt == 'SGD' then
    _, fs = optim.sgd(feval, x, params)
  elseif global_args.opt == 'Adam' then
    _, fs = optim.adam(feval, x, params)
  end
  -- fs is a table containing value of the loss function
  -- (just 1 value for the SGD optimization)
  
  count = count + 1
  current_loss = current_loss + fs[1]
  
  -- normalize loss
  return current_loss / count
end