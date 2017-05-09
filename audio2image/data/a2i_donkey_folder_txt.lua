auxdata_aux1require 'image'
dir = require 'pl.dir'

trainLoader = {}

--[[
local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
  dict[alphabet:sub(i,i)] = i -- Substring : mapping letters to indices
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k -- Mapping indices to leters
end
alphabet_size = #alphabet

-- Given encoded txt, decode to text
function decode(txt)
  local str = ''
  for w_ix = 1,txt:size(1) do
    local ch_ix = txt[w_ix]
    local ch = ivocab[ch_ix]
    if (ch  ~= nil) then
      str = str .. ch
    end
  end
  return str
end

function trainLoader:decode2(txt)
  local str = ''
  _, ch_ixs = txt:max(2)
  for w_ix = 1,txt:size(1) do
    local ch_ix = ch_ixs[{w_ix,1}]
    local ch = ivocab[ch_ix]
    if (ch ~= nil) then
      str = str .. ch
    end
  end
  return str
end

trainLoader.alphabet = alphabet
trainLoader.alphabet_size = alphabet_size
trainLoader.dict = dict
trainLoader.ivocab = ivocab
trainLoader.decode = decoder

--]]

local classnames = {}
for line in io.lines(opt.classnames) do
  classnames[#classnames + 1] = line
end

local files = {}
local trainids = {}
local size = 0
for line in io.lines(opt.trainids) do
  local id = tonumber(line)
  local dirpath = opt.data_root .. '/' .. classnames[id]
  cur_files = dir.getfiles(dirpath)
  files[id] = cur_files
  size = size + #cur_files
  trainids[#trainids + 1] = id
end

--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, loadSize[2], loadSize[2])
  return input
end

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(path)
  collectgarbage()
  local input = loadImage(path)

  -- If no_aug, we just scale the image down
  if opt.no_aug == 1 then
    return image.scale(input, sampleSize[2], sampleSize[2])
  end

  local iW = input:size(3)
  local iH = input:size(2)

  -- do random crop
  local oW = sampleSize[2];
  local oH = sampleSize[2]
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
  assert(out:size(2) == oW)
  assert(out:size(3) == oH)
  -- do hflip with probability 0.5
  if torch.uniform() > 0.5 then out = image.hflip(out); end
  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
  return out
end

function trainLoader:sample(quantity)
  if opt.replicate == 1 then
    return self:sample_repl(quantity)
  else
    return self:sample_no_repl(quantity)
  end
end

-- quantity : batchsize
function trainLoader:sample_no_repl(quantity)
  local ix_batch1 = torch.Tensor(quantity)
  local ix_batch2 = torch.Tensor(quantity)
  local ix_file1 = torch.Tensor(quantity)
  local ix_file2 = torch.Tensor(quantity)

  -- Share image / text among adjacent opt.numAudio data samples.
  for n = 1, quantity do
    local cls_ix = torch.randperm(#trainids):narrow(1,1,2) -- select two random classes
    local id1 = trainids[cls_ix[1]] -- id of class 1
    local id2 = trainids[cls_ix[2]] -- id of class 2
    local file_ix1 = torch.randperm(#files[id1])[1] -- choose a random file class 1
    local file_ix2 = torch.randperm(#files[id2])[1] -- choose a random file in class 2
    ix_batch1[n] = cls_ix[1] -- Include 1st class id in 1st batch
    ix_batch2[n] = cls_ix[2] -- Include 2nd class id in 2nd batch
    ix_file1[n] = file_ix1 -- Include 1st file in 1st batch of files
    ix_file2[n] = file_ix2 -- Include 2nd file in 2nd batch of files
  end

  local data_img1 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- real
  local data_img2 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- wrong
  local data_aux1 = torch.zeros(quantity*opt.numAudio, opt.auxSize) -- real
  local ids = torch.zeros(quantity)

  local aux_batch_ix = 1
  for n = 1, quantity do
    local id1 = trainids[ix_batch1[n]] -- get nth id of 1st batch
    local id2 = trainids[ix_batch2[n]] -- get nth id of 2nd batch
    ids[n] = id1                        -- save the index of 1st batch into 'ids'
    local cls1_files = files[id1]      -- get files of nth clss of 1st batch
    local cls2_files = files[id2]     -- get files of nth class of 2nd batch

    local t7file1 = cls1_files[ix_file1[n]] -- Get this file from 1st class
    local t7file2 = cls2_files[ix_file2[n]] -- Get this file from 2nd class
    local info1 = torch.load(t7file1)
    local info2 = torch.load(t7file2)

    local img_file1 = opt.img_dir .. '/' .. info1.img_name   -- Load the image from image diretory
    local img1 = trainHook(img_file1)
    local img_file2 = opt.img_dir .. '/' .. info2.img_name
    local img2 = trainHook(img_file2)

    for s = 1, opt.numAudio do
      local ix_aux1 = torch.randperm(info1[1]:t():size(1))[1]
      data_aux1[aux_batch_ix]:copy(info1[1]:t()[ix_aux1])
      aux_batch_ix = aux_batch_ix + 1
    end

    data_img1[n]:copy(img1)
    data_img2[n]:copy(img2)
  end
  collectgarbage(); collectgarbage()
  return data_img1, data_aux1, data_img2, ids
end

function trainLoader:sample_repl(quantity)
  local numfile = math.floor(quantity / opt.numAudio)
  assert(numfile * opt.numAudio == quantity)

  local ix_batch1 = torch.Tensor(quantity)
  local ix_batch2 = torch.Tensor(quantity)
  local ix_file1 = torch.Tensor(quantity)
  local ix_file2 = torch.Tensor(quantity)
  local n = 1

  -- Share image / text among adjacent opt.numAudio data samples.
  for f = 1, numfile do
    local cls_ix = torch.randperm(#trainids):narrow(1,1,2)
    local id1 = trainids[cls_ix[1]]
    local id2 = trainids[cls_ix[2]]
    local file_ix1 = torch.randperm(#files[id1])[1]
    local file_ix2 = torch.randperm(#files[id2])[1]
    for s = 1, opt.numAudio do
      ix_batch1[n] = cls_ix[1]
      ix_batch2[n] = cls_ix[2]
      ix_file1[n] = file_ix1
      ix_file2[n] = file_ix2
      n = n + 1
    end
  end

  local data_img1 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- real
  local data_img2 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- wrong
  local data_aux1 = torch.zeros(quantity, opt.auxSize) -- real
  local ids = torch.zeros(quantity)

  for n = 1, quantity do
    local id1 = trainids[ix_batch1[n]]
    local id2 = trainids[ix_batch2[n]]
    ids[n] = id1
    local cls1_files = files[id1]
    local cls2_files = files[id2]

    local t7file1 = cls1_files[ix_file1[n]]
    local t7file2 = cls2_files[ix_file2[n]]
    local info1 = torch.load(t7file1)
    local info2 = torch.load(t7file2)

    local img_file1 = opt.img_dir .. '/' .. info1.img_name
    local img1 = trainHook(img_file1)
    local img_file2 = opt.img_dir .. '/' .. info2.img_name
    local img2 = trainHook(img_file2)
    local ix_aux1 = torch.randperm(info1[1]:t():size(1))[1]

    local aux1 = info1[1]:t()[ix_aux1]
    data_aux1[n]:copy(aux1)
    data_img1[n]:copy(img1)
    data_img2[n]:copy(img2)
  end
  collectgarbage(); collectgarbage()
  return data_img1, data_aux1, data_img2, ids
end

function trainLoader:size()
  return size
end
