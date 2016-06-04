require 'Trainer.ObserverPattern'

require 'xlua'
local View_Progress = torch.class('View_Progress','Observer')

function View_Progress:__init(args)
  Observer:__init(args)
end

function View_Progress:update(Observable,Msg)
  local current = Observable.sampler.current_train_batch
  local full = Observable.sampler.n_train_batches
  
  xlua.progress(current,full)
end