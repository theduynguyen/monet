require 'torch'

local Preprocessor_Base = torch.class('Preprocessor_Base')

function Preprocessor_Base:__init(args)
  self.args = args
end

function Preprocessor_Base:process(dataset)
  error('Abstract class')
end