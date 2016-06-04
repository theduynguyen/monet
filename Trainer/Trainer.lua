require 'torch'

require 'Trainer.StoppingCrit'
require 'Trainer.ObserverPattern'

local Trainer = torch.class('Trainer','Observable')

-- Template method - need to specify blocks
function Trainer:__init(args,components)
  Observable:__init(args)
  
  self.model = components.model
  self.sampler = components.sampler
  self.modelUpdater = components.modelUpdater
  
  self.stoppingCrit = components.stoppingCrit or { StoppingCrit_MaxIter({}) }
  self.current_epoch = 1
  self.current_train_loss = -1
  
  
  -- create notification messages
  -- Start training
  self:create_msg('Start')
  
  -- Batch finished
  self:create_msg('Batch')
  
  -- Epoch finished
  self:create_msg('Epoch')
  
  -- Training finished
  self:create_msg('Stop')
end

function Trainer:do_epoch()
  print('Epoch '..self.current_epoch)
  
  --initialise sampler
  self.sampler:new_epoch()
  local epoch_loss = 0
  local batch_count = 1
  
  -- for each epoch
  repeat
    -- sample one batch from sampler
    local batch = self.sampler:get_train_batch()
    
    -- do a forward-backward pass
    epoch_loss = epoch_loss + self.modelUpdater:step(self.model,batch)
    batch_count = batch_count + 1
    
    self:notify('Batch')
    
  until( self.sampler:is_full_epoch() == true )
  
  self.current_train_loss = epoch_loss / batch_count  
  self:notify('Epoch')
end

function Trainer:train()
  local stop = false
  self:notify('Start')
  
  repeat
    self:do_epoch()
    
    for k, v in pairs(self.stoppingCrit) do
      stop = stop or v:check(self)
    end
    
    self.current_epoch = self.current_epoch + 1   
  until(stop == true)
  
  self:notify('Stop')
end
