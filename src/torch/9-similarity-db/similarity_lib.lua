-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the Apache License v2.0 (http://www.apache.org/licenses/LICENSE-2.0.txt).

require 'torch'

local similarity_lib = {}

function similarity_lib.similarity(referenceEncoding, imageEncoding)
  local sumSimilarity = 0.0
  local minHeight = math.min(referenceEncoding:size()[2], imageEncoding:size()[2])
  local maxHeight = math.max(referenceEncoding:size()[2], imageEncoding:size()[2])
  if (maxHeight - minHeight < 5) then
    for w = 1, referenceEncoding:size()[1] do
      for h = 1, minHeight do
        local similarityLoc = imageEncoding[w][h] * referenceEncoding[w][h]
        sumSimilarity = sumSimilarity + similarityLoc
      end
    end
    local similarity = sumSimilarity / (referenceEncoding:size()[1] * minHeight)
    return similarity
  else
    return -1.0
  end
end

return similarity_lib
