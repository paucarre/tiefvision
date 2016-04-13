-- Copyright (C) 2016 Pau Carré Cardona - All Rights Reserved
-- You may use, distribute and modify this code under the
-- terms of the GPL v2 license (http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt).

local image = require 'image'

function detectGrayscale(imagePath)
  local img = image.load(imagePath)
  if img:size()[1] == 3 then
    print("C")
  else
    print("G")
  end
end

detectGrayscale(arg[1])
