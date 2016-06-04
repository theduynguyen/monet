require 'torch'
require 'hypero'

require 'View.View_Accuracy'
local View_Hypero = torch.class('View_Hypero','Observer')

function View_Hypero:__init(args)
  Observer:__init(args)
  
  self.bat_name = args.bat_name or 'Test'
  self.version = args.version or 'v1'
  
  -- connect to hypero server
  conn = hypero.connect()

  local batName = self.bat_name
  local verDesc = self.version
  self.battery = conn:battery(batName, verDesc)
  
  -- hyperparams
  self.hp = {}
  self.hp.epochs = global_args.epochs
  self.hp.batch_size = global_args.batch_size
  
  -- results
  self.res = {}
  
  -- meta data
  self.md = {}
end

function View_Hypero:update(Observable,Msg)
  if Msg == 'Start' then
    
    -- write model
    self.md.modelstr = tostring(Observable.model.model)
  end
  
  if Msg == 'Epoch' then
    -- log train loss, test loss and test acc for each epoch
  end

  if Msg == 'Stop' then
    -- compute and save result
    local v_acc = View_Accuracy({show_progress = false})
    local loss, acc = v_acc:compute_test_measures(Observable)
    self.res.train_loss = Observable.current_train_loss
    self.res.test_loss = loss
    self.res.test_acc = acc
    
    -- save experiment
    local hex = self.battery:experiment()
    hex:setParam(self.hp)
    hex:setResult(self.res)
    hex:setMeta(self.md)
  end
end