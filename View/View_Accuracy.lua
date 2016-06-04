require 'torch'
require 'Trainer.ObserverPattern'

require 'xlua'
local View_Accuracy = torch.class('View_Accuracy','Observer')

function View_Accuracy:__init(args)
  Observer:__init(args)
  
  self.show_progress = args.show_progress or true
end

function View_Accuracy:update(Observable,Msg)
  local loss, acc = self:compute_test_measures(Observable)
  
  print('Train loss: '..Observable.current_train_loss)
  print('Test loss: '..loss)
  print('Test accuracy: '..acc)
end

function View_Accuracy:compute_test_measures(Observable)
  local count = 0
  local current_loss = 0
  local batch_count = 0
  
  -- initialise test epochs
  Observable.sampler:new_epoch()
  local i = 0
  
  repeat
      local batch = Observable.sampler:get_test_batch()
      local outputs = Observable.model.model:forward(batch.inputs)
      
      local _, indices = torch.max(outputs:float(), 2)
      local guessed_right = indices:long():eq(batch.targets:long()):sum()
      
      count = count + guessed_right
      
      local loss = Observable.model.criterion:forward(outputs, batch.targets)
      current_loss = current_loss + loss
      
      if self.show_progress == true then
        local current = Observable.sampler.current_test_batch
        local full = Observable.sampler.n_test_batches
    
        xlua.progress(current,full)
      end
  until( Observable.sampler:is_full_epoch() == true )
  
  local acc = count / Observable.sampler.dataset.test_size
  local loss = current_loss / Observable.sampler.n_test_batches
  
  return loss, acc
end