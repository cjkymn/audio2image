--[[
-- Modified from : Generative Adversarial Text-to-Image Synthesis
   Original source : https://github.com/reedscot/icml2016
   Author : MinHyung Kang, Chris Kymn

   Generic training script for GAN, GAN-CLS, GAN-INT, GAN-CLS-INT.
   Training script for audio-to-image synthesis
--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'cudnn'

opt = {
   numAudio = 1, -- Number of audio samples per image
   replicate = 0, -- TODO : Could change this (if 1, then replicate averaged text features numAudio times.)
   save_every = 1,
   print_every = 1,
   dataset = 'instruments',       -- imagenet / lsun / folder
   no_aug = 0,
   img_dir = '/idata/cs189/data/audio2image/data/instruments/ucfframes',
   keep_img_frac = 1.0,
   interp_weight = 0,
   interp_type = 1,
   cls_weight = 0,
   filenames = '',
   data_root = '/idata/cs189/data/audio2image/data/instruments/ucfaudionorm',
   classnames = '/idata/cs189/data/audio2image/data/instruments/allclasses.txt',
   trainids = '/idata/cs189/data/audio2image/data/instruments/allids.txt',
   checkpoint_dir = '/idata/cs189/data/audio2image/checkpoints',
   numshot = 0,
   batchSize = 64,
   doc_length = 201,
   loadSize = 76,
   auxSize = 1024,         -- #  of dim for raw text.
   fineSize = 64,
   na = 128,               -- #  of dim for audio features.
   nz = 100,               -- #  of dim for Z
   ngf = 128,              -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 1000,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   lr_decay = 0.5,            -- initial learning rate for adam
   decay_every = 100,
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 2,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'normalized_vec',
   noise = 'normal',       -- uniform / normal
   init_g = '',
   init_d = '',
   use_cudnn = 1,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if opt.gpu > 0 then
   ok, cunn = pcall(require, 'cunn')
   ok2, cutorch = pcall(require, 'cutorch')
   cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/a2i_data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------

-- Init weights (convolutional layers vs batch normalization)
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz -- #  of dim for Z
local ndf = opt.ndf -- #  of discrim filters in first conv layer
local ngf = opt.ngf -- #  of gen filters in first conv layer
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution


-- Create generator network
if opt.init_g == '' then
    -- TODO : should be changed to sound features
  fcG = nn.Sequential()
  fcG:add(nn.Linear(opt.auxSize,opt.na)) -- Text to text features
  fcG:add(nn.LeakyReLU(0.2,true)) -- nonlinearity
  netG = nn.Sequential()
  -- concat Z and txt
  ptg = nn.ParallelTable()
  ptg:add(nn.Identity()) -- for noise
  ptg:add(fcG)
  netG:add(ptg) -- ptg contains : 1 identity, 1 fcG (Text to features)
  netG:add(nn.JoinTable(2)) -- Concatenates result along 2nd dimension
  -- input is Z, going into a convolution
  netG:add(SpatialFullConvolution(nz + opt.na, ngf * 8, 4, 4)) -- Kernel width, height = 4,4
  netG:add(SpatialBatchNormalization(ngf * 8))

    -- state size: (ngf*8) x 4 x 4
    local conc = nn.ConcatTable() -- Each member module to same input
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ngf * 8, ngf * 2, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
    conv:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf * 2))

    conv:add(nn.ReLU(true))
    conv:add(SpatialConvolution(ngf * 2, ngf * 8, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf * 8))
    conc:add(nn.Identity())
    conc:add(conv)
    netG:add(conc)
    netG:add(nn.CAddTable())
    netG:add(nn.ReLU(true))

  -- state size: (ngf*8) x 4 x c
  netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(ngf * 4))

    -- state size: (ngf*4) x 8 x 8
    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ngf * 4, ngf, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
    conv:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf))

    conv:add(nn.ReLU(true))
    conv:add(SpatialConvolution(ngf, ngf * 4, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ngf * 4))
    conc:add(nn.Identity())
    conc:add(conv)
    netG:add(conc)
    netG:add(nn.CAddTable())
    netG:add(nn.ReLU(true))

      -- state size: (ngf*4) x 8 x 8
  netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))

  -- state size: (ngf*2) x 16 x 16
  netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

  -- state size: (ngf) x 32 x 32
  netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
  netG:add(nn.Tanh())

  -- state size: (nc) x 64 x 64
  netG:apply(weights_init)
else -- There is already initialized G
  netG = torch.load(opt.init_g)
end

-- Discriminator
if opt.init_d == '' then
  convD = nn.Sequential()
  -- input is (nc) x 64 x 64 : Here it's 3 : probably rgb?
  convD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
  convD:add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf) x 32 x 32
  convD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
  convD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*2) x 16 x 16
  convD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
  convD:add(SpatialBatchNormalization(ndf * 4))

  -- state size: (ndf*4) x 8 x 8
  convD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
  convD:add(SpatialBatchNormalization(ndf * 8))

    -- state size: (ndf*8) x 4 x 4
    local conc = nn.ConcatTable()
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(ndf * 8, ndf * 2, 1, 1, 1, 1, 0, 0))
    conv:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size : (ndf * 2) x 4 x 4
    conv:add(SpatialConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 2))
    conv:add(nn.LeakyReLU(0.2, true))
    -- state size : (ndf * 2) x 4 x 4
    conv:add(SpatialConvolution(ndf * 2, ndf * 8, 3, 3, 1, 1, 1, 1))
    conv:add(SpatialBatchNormalization(ndf * 8))
    -- state size : (ndf * 8) x 4 x 4
    conc:add(nn.Identity())
    conc:add(conv)
    convD:add(conc)
    convD:add(nn.CAddTable())
    convD:add(nn.LeakyReLU(0.2, true))

    -- TODO : Takes input as text (should be changed to sound)
  local fcD = nn.Sequential()
  fcD:add(nn.Linear(opt.auxSize,opt.na))
  fcD:add(nn.BatchNormalization(opt.na))
  fcD:add(nn.LeakyReLU(0.2,true))
  fcD:add(nn.Replicate(4,3))
  fcD:add(nn.Replicate(4,4))
  netD = nn.Sequential()
  pt = nn.ParallelTable()
  pt:add(convD)
  pt:add(fcD)
  netD:add(pt)
  netD:add(nn.JoinTable(2))
  -- state size: (ndf*8 + 128) x 4 x 4
  netD:add(SpatialConvolution(ndf * 8 + opt.na, ndf * 8, 1, 1)) -- filters of (Text + features from img) combined
  netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf*8) x 4 x 4
  netD:add(SpatialConvolution(ndf * 8, 1, 4, 4)) -- One final layer
  netD:add(nn.Sigmoid()) -- Scalar, probability
  -- state size: 1 x 1 x 1
  netD:add(nn.View(1):setNumInputDims(3))
  -- state size: 1
  netD:apply(weights_init)
else -- pretrained model
  netD = torch.load(opt.init_d)
end
-- Check batchsize is divisible by number of captions.
assert(math.floor(opt.batchSize / opt.numAudio) * opt.numAudio == opt.batchSize)
netR = nn.Sequential()
if opt.replicate == 1 then -- Hmm? Take average of opt.numAudio amount in batches, repeat?
  netR:add(nn.Reshape(opt.batchSize / opt.numAudio, opt.numAudio, opt.auxSize))
  netR:add(nn.Transpose({1,2}))
  netR:add(nn.Mean(1))
  netR:add(nn.Replicate(opt.numAudio))
  netR:add(nn.Transpose({1,2}))
  netR:add(nn.Reshape(opt.batchSize, opt.auxSize))
else
  netR:add(nn.Reshape(opt.batchSize, opt.numAudio, opt.auxSize))
  netR:add(nn.Transpose({1,2}))
  netR:add(nn.Mean(1))
end

local criterion = nn.BCECriterion()
local weights = torch.zeros(opt.batchSize * 3/2)
weights:narrow(1,1,opt.batchSize):fill(1)
weights:narrow(1,opt.batchSize+1,opt.batchSize/2):fill(opt.interp_weight)
local criterion_interp = nn.BCECriterion(weights)
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
--[[
local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
alphabet_size = #alphabet
--]]
local input_img = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- Interpolation
local input_img_interp = torch.Tensor(opt.batchSize * 3/2, 3, opt.fineSize, opt.fineSize)
if opt.replicate == 1 then
  input_txt_raw = torch.Tensor(opt.batchSize, opt.auxSize)
else
  input_txt_raw = torch.Tensor(opt.batchSize * opt.numAudio, opt.auxSize)
end
local input_txt = torch.Tensor(opt.batchSize, opt.auxSize)
local input_txt_interp = torch.zeros(opt.batchSize * 3/2, opt.auxSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local noise_interp = torch.Tensor(opt.batchSize * 3/2, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local label_interp = torch.Tensor(opt.batchSize * 3/2)
local stats = torch.zeros(math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)*opt.niter,6)
local errD, errG, errW
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_img_interp = input_img_interp:cuda()
   input_txt = input_txt:cuda()
   input_txt_raw = input_txt_raw:cuda()
   input_txt_interp = input_txt_interp:cuda()
   noise = noise:cuda()
   noise_interp = noise_interp:cuda()
   label = label:cuda()
   label_interp = label_interp:cuda()
   netD:cuda()
   netG:cuda()
   netR:cuda()
   criterion:cuda()
   criterion_interp:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netD = cudnn.convert(netD, cudnn)
  netG = cudnn.convert(netG, cudnn)
  netR = cudnn.convert(netR, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersD:zero()

  -- train with real
  data_tm:reset(); data_tm:resume()
  -- TODO : what is getBatch doing??????
  real_img, real_txt, wrong_img, _ = data:getBatch()
  data_tm:stop()

  input_img:copy(real_img)
  input_txt_raw:copy(real_txt)
  -- average adjacent text features in batch dimension.
  emb_txt = netR:forward(input_txt_raw)
  input_txt:copy(emb_txt)

  if opt.interp_type == 1 then -- This works well
    -- compute (a + b)/2
    input_txt_interp:narrow(1,1,opt.batchSize):copy(input_txt)
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):copy(input_txt:narrow(1,1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(input_txt:narrow(1,opt.batchSize/2+1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):mul(0.5)
  elseif opt.interp_type == 2 then
    -- compute (a + b)/2
    input_txt_interp:narrow(1,1,opt.batchSize):copy(input_txt)
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):copy(input_txt:narrow(1,1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(input_txt:narrow(1,opt.batchSize/2+1,opt.batchSize/2))
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):mul(0.5)

    -- add extrapolation vector.
    local alpha = torch.rand(opt.batchSize/2,1):mul(2):add(-1) -- alpha ~ uniform(-1,1)
    if opt.gpu >=0 then
     alpha = alpha:float():cuda()
    end
    alpha = torch.expand(alpha,opt.batchSize/2,input_txt_interp:size(2))
    local vec = (input_txt:narrow(1,opt.batchSize/2+1,opt.batchSize/2) -
                input_txt:narrow(1,1,opt.batchSize/2)):cmul(alpha) -- elementwise multiplication
    input_txt_interp:narrow(1,opt.batchSize+1,opt.batchSize/2):add(vec)
  end
  label:fill(real_label)

  local output = netD:forward{input_img, input_txt}
  local errD_real = criterion:forward(output, label) -- error of real label
  local df_do = criterion:backward(output, label) -- backproapagation
  netD:backward({input_img, input_txt}, df_do)

  errD_wrong = 0
  if opt.cls_weight > 0 then
    -- train with wrong image-txt pair
    input_img:copy(wrong_img)
    label:fill(fake_label)

    local output = netD:forward{input_img, input_txt}
    errD_wrong = opt.cls_weight*criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    df_do:mul(opt.cls_weight)
    netD:backward({input_img, input_txt}, df_do)
  end

  -- train with fake
  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end
  local fake = netG:forward{noise, input_txt}
  input_img:copy(fake)
  label:fill(fake_label)

  local output = netD:forward{input_img, input_txt}
  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local fake_weight = 1 - opt.cls_weight
  errD_fake = errD_fake*fake_weight
  df_do:mul(fake_weight)
  netD:backward({input_img, input_txt}, df_do)

  errD = errD_real + errD_fake + errD_wrong
  errW = errD_wrong

  return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
  netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
  netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

  gradParametersG:zero()

  if opt.noise == 'uniform' then -- regenerate random noise
    noise_interp:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise_interp:normal(0, 1)
  end
  local fake = netG:forward{noise_interp, input_txt_interp}
  input_img_interp:copy(fake)
  label_interp:fill(real_label) -- fake labels are real for generator cost

  local output = netD:forward{input_img_interp, input_txt_interp}
  errG = criterion_interp:forward(output, label_interp)
  local df_do = criterion_interp:backward(output, label_interp)
  local df_dg = netD:updateGradInput({input_img_interp, input_txt_interp}, df_do)

  netG:backward({noise_interp, input_txt_interp}, df_dg[1])
  return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
  epoch_tm:reset()

  if epoch % opt.decay_every == 0 then
    optimStateG.learningRate = optimStateG.learningRate * opt.lr_decay
    optimStateD.learningRate = optimStateD.learningRate * opt.lr_decay
  end
  local statIndex = 1

  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    tm:reset()
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    optim.adam(fDx, parametersD, optimStateD)

    -- (2) Update G network: maximize log(D(G(z)))
    optim.adam(fGx, parametersG, optimStateG)

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
        printString = ('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                  .. '  Err_G: %.4f  Err_D: %.4f Err_W: %.4f'):format(
                epoch, ((i-1) / opt.batchSize),
                math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                tm:time().real, data_tm:time().real,
                optimStateG.learningRate,
                errG and errG or -1, errD and errD or -1,
                errW and errW or -1)
        print(printString)
        local iterStep = ((i-1) / opt.batchSize)
        stats[statIndex][1] = epoch
        stats[statIndex][2] = iterStep
        stats[statIndex][3] = optimStateG.learningRate
        stats[statIndex][4] = errG and errG or -1
        stats[statIndex][5] = errD and errD or -1
        stats[statIndex][6] = errW and errW or -1

        statIndex = statIndex + 1

      local fake = netG.output
      disp.image(fake:narrow(1,1,opt.batchSize), {win=opt.display_id, title=opt.name})
      disp.image(real_img, {win=opt.display_id * 3, title=opt.name})
    end
  end
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)

    print('Saving at ' ..opt.checkpoint_dir)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_stats.t7',stats)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end
