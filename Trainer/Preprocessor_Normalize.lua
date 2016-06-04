require 'Trainer.Preprocessor_Base'

local Model_CNN = torch.class('Preprocessor_Normalize','Preprocessor_Base')

function Preprocessor_Normalize:__init(args)
  Preprocessor_Base:__init(args)
end

function Preprocessor_Normalize:process(dataset)
  local m_std = self.args.std or dataset:std()
  local m_mean = self.args.mean or dataset:mean()
  
  dataset:add(-m_mean)
  dataset:div(m_std)
  return dataset, m_mean, m_std
end
