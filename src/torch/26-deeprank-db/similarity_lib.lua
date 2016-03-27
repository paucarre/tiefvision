-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

require 'torch'

local similarity_lib = {}

function similarity_lib.similarity(referenceEncoding, imageEncoding)
  local sumSimilarity = 0.0
  if(referenceEncoding:size():size() == 3 and imageEncoding:size():size() == 3) then
    --print(referenceEncoding:size())
    --print(imageEncoding:size())
    referenceEncoding = referenceEncoding:transpose(1, 3)
    imageEncoding = imageEncoding:transpose(1, 3)
    local minHeight = math.min(referenceEncoding:size()[2], imageEncoding:size()[2])
    local maxHeight = math.max(referenceEncoding:size()[2], imageEncoding:size()[2])
    if( maxHeight - minHeight < 5) then
      for h = 1, minHeight do
        for w = 1, referenceEncoding:size()[1] do
          local similarityLoc = imageEncoding[w][h] * referenceEncoding[w][h]
          sumSimilarity =  sumSimilarity + similarityLoc
        end
      end
      local similarity = sumSimilarity / (minHeight * referenceEncoding:size()[1])
      return similarity
    end
  end
  return -1.0
end

return similarity_lib
