require 'torch'

local StoppingCrit_Base = torch.class('StoppingCrit_Base')

function StoppingCrit_Base:__init(args)
  self.args = args
end

function StoppingCrit_Base:check(trainer)
end

-- maximum iterations
local StoppingCrit_MaxIter = torch.class('StoppingCrit_MaxIter','StoppingCrit_Base')

function StoppingCrit_MaxIter:__init(args)
  StoppingCrit_Base:__init(args)
  
  self.max_epoch = args.max_epoch or 5
end

function StoppingCrit_MaxIter:check(trainer)
  
  if trainer.current_epoch >= self.max_epoch then
    return true
  else
    return false
  end
end