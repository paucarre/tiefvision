-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

require "inn"
require 'optim'
require 'torch'
require 'xlua'

local classifierconv = {}

function classifierconv.loadModel()
   local nhiddens1 = 1024
   local nhiddens2 = 128 -- 256
   local noutputs = 16
   local model = nn.Sequential()
   model:add(nn.SpatialConvolutionMM(384, nhiddens1, 11, 11, 1, 1, 0, 0))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolutionMM(nhiddens1, nhiddens2, 1, 1, 1, 1, 0, 0))
   model:add(nn.Sigmoid()) -- encoding used for similarity
   -- model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolutionMM(nhiddens2, noutputs, 1, 1, 1, 1, 0, 0))
   model:add(nn.Sigmoid())
   return model:cuda()
end

return classifierconv
