require 'torch'

--implements Observer pattern with different messages

local Observable = torch.class('Observable')

function Observable:__init(args)
  self.args = args
  
  self.Observers = {}
end

function Observable:create_msg(Msg)
  self.Observers[Msg] = {}
end

function Observable:register(Msg,Observer)
  table.insert(self.Observers[Msg],Observer)
end

function Observable:notify(Msg)
  for k, v in pairs(self.Observers[Msg]) do
    v:update(self,Msg)
  end
end


local Observer = torch.class('Observer')

function Observer:__init(args)
  self.args = args
end

function Observer:update(Observable,Msg)
end
