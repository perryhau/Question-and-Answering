--------------------------------------------------------------------------------------------------------------------------------------
-- Name:        qa_lstm.lua
-- Description: QA-LSTM(maxpool) module from paper "Improved Representation Learning for Question Answer Matching" by Ming et al 2016
-- Written by:  Stalin Varanasi
-- Remarks:     read_qa.lua is used to load questions,answers of training and validation set ( files should be of the format mentioned in :  https://github.com/shuzi/insuranceQA.git) 
--              Write your own read_qacorpus.lua file to fit corpora of other formats.
---------------------------------------------------------------------------------------------------------------------------------------



require('nngraph')
require('nn')
require('rnn')
require('./read_qacorpus')


require 'cutorch/init'
require 'cunn/init'



local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        LookupTable = nn.LookupTableMaskZero
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTableMaskZero
end

cutorch.setDevice(1)
print("Number of gpus="..cutorch.getDeviceCount())
print("Using gpu with id="..cutorch.getDevice())


local DATA_DIR = "../../"

local params = {
                batch_size=20, -- number of qa-pairs per one forward-backward
                layers=1,      -- decides how many layers of LSTMs we want to use
                wvec_size=100,
                rnn_size=141,
                dropout=0.5,
                lr=1.1,
                vocab_size=10000,
                max_epoch=30,
                margin=0.2,
                anspool=50,
                maxseqlen=200 }



local function transfer_data(x)
  return x:cuda()
end



local function create_modules()

qatolstm=nn.FastLSTM(params.wvec_size,params.rnn_size):maskZero(1)
qafrolstm=nn.FastLSTM(params.wvec_size,params.rnn_size):maskZero(1)
lookuptable_to=nn.LookupTableMaskZero(params.vocab_size,params.wvec_size)
lookuptable_fro=lookuptable_to:clone('weight','bias')

local qato_unit=nn.Sequential()
qato_unit:add(lookuptable_to)
qato_unit:add(qatolstm)
qato_unit:add(nn.MaskZero(nn.AddConstant(100),1))
qato_unit:add(nn.AddConstant(-100))
qato_unit:add(nn.Reshape(params.rnn_size,1)) -- necessary to merge finally

local qafro_unit=nn.Sequential()
qafro_unit:add(lookuptable_fro)
qafro_unit:add(qafrolstm)
qafro_unit:add(nn.MaskZero(nn.AddConstant(100),1))
qafro_unit:add(nn.AddConstant(-100))
qafro_unit:add(nn.Reshape(params.rnn_size,1)) -- necessary to merge finally

local qamod=nn.BiSequencer(qato_unit,qafro_unit,2)


--maxcosmodule
local q=nn.Identity()()
local a=nn.Identity()()
local wa=nn.Identity()()

local qvec=nn.Max(3)(nn.JoinTable(3)(q))
local avec=nn.Max(3)(nn.JoinTable(3)(a))
local wavec=nn.Max(3)(nn.JoinTable(3)(wa))

local cosqa=nn.CosineDistance()(nn.Identity()({qvec,avec}))
local cosqwa=nn.CosineDistance()(nn.Identity()({qvec,wavec}))

local maxcosmodule=transfer_data(nn.gModule({q,a,wa},{cosqa,cosqwa}))

return transfer_data(qamod),maxcosmodule 

end



local function setup()

print("#Creating a bi-directional RNN(LSTM) network..")

if ( not( tonumber(arg[1])>=1 and tonumber(arg[1])<=4 ) ) then

   qmod,maxcosmod=create_modules()
   -- Sharing LookupTable for both to and fro RNN
   qmod:get(1):get(1):get(2):get(2):get(1):get(1):get(1):share(qmod:get(1):get(1):get(1):get(1):get(1):get(1),'weight','bias')

   amod=qmod:clone('weight','bias')
   wamod=amod:clone('weight','bias')
   joinmod=nn.JoinTable(3):cuda()

   --Add WordVectors
   print("#Initializing LookupTable with word vectors..")
   local embeddingsFile,msg = io.open(string.format('%s/data/LiveQA/Words2Vectors',DATA_DIR),"r")
   local all=embeddingsFile:read("*all")
   vectable={}
   for line in string.gmatch(all,"([^\n]*)\n") do
        local tmptbl={}
        local idbegin,idend = string.find(line,"[%s]+",1)
        local word = string.sub(line,1,idbegin-1)
        local vecline = string.sub(line,idend+1)

        for value in string.gmatch(vecline, "[^%s]+") do
            table.insert(tmptbl,tonumber(value))
        end
        vectable[word]=torch.Tensor(tmptbl)
   end


  -- Update the weights(vector) for respective words in LookupTable.
   for idx,word in pairs(vocab) do
     local num,count = string.gsub(idx,"idx_","") 
     if ( vectable[word] ~= nil  ) then
       qmod:get(1):get(1):get(1):get(1):get(1):get(1).weight[tonumber(num)]:copy(vectable[word])
       --print("word "..word..","..num.." is in the word2vec file")
     end
   end



print("->Created the network.")

else -- first argument is between 1-to-4

  cutorch.setDevice(tonumber(arg[1]))
  qmod=torch.load('./qamod.best.net')
  amod=qmod:clone('weight','bias')
  wamod=amod:clone('weight','bias')
  qmod.maskzero=true
  amod.maskzero=true
  wamod.maskzero=true
  qmod:training()
  amod:training()
  wamod:training()
  _,maxcosmod=create_modules()

print("->Loaded the  network")
end

end




          
local function texts2matrix(texts,maxlen)
     if ( maxlen > params.maxseqlen ) then
         maxlen = params.maxseqlen 
     end
     local tmatrix=transfer_data(torch.zeros(#texts,maxlen))
     local maskmatrix=transfer_data(torch.ones(#texts,2*params.rnn_size,maxlen)*-999)
        
             
     local tcounts={}

     for textid=1,#texts do

     if ( texts[textid]:size(1) < maxlen ) then
       tmatrix[textid]:narrow(1,1,texts[textid]:size(1)):copy(texts[textid])
       maskmatrix[textid]:narrow(2,1,texts[textid]:size(1)):fill(0)
     else
       tmatrix[textid]:copy(texts[textid]:narrow(1,1,maxlen))
       maskmatrix[textid]:narrow(2,1,maxlen):fill(0)
     end
     
     table.insert(tcounts,texts[textid]:size(1))

     end
     local tmatrix_t=tmatrix:t()
     local tmattbl={}
     for i=1,tmatrix_t:size(1) do
          table.insert(tmattbl,tmatrix_t[i])
     end
     return tmattbl,maskmatrix
  
end


local function check_valid()
     local cosinedist=transfer_data(nn.CosineDistance())

     -- Assign vectors to all questions and answers 
     local avecmatrix=transfer_data(torch.zeros(#answerdict,2*params.rnn_size))
     print("#Generating vectors for all answers..(can take few minutes)")
     for i=1,#answerdict,500 do
           local texts={}
           local maxlen=0
           local stepsize=500 -- This defines the batch-size while generating vectors for all answers
            if ( i+stepsize > #answerdict ) then
                   stepsize=#answerdict-i+1
            end
           for j=0,stepsize-1 do
                table.insert(texts,answerdict[j+i]) 
                if ( answerdict[j+i]:size(1) > maxlen ) then
                     maxlen=answerdict[j+i]:size(1)
                end
           end

           local amatrix,amask=texts2matrix(texts,maxlen)
           avecmatrix:narrow(1,i,stepsize):copy(torch.max(nn.JoinTable(3):cuda():forward(avecmod:forward(amatrix)),3)) -- amask will make the filler's output invalid while taking torch.max
     end


     local id=1
 
     local amatrix=transfer_data(torch.zeros(qpool[id]:size(1),2*params.rnn_size)) 
     local reciprocalrank=torch.zeros(#qdev)
     local accuracy=0
     while id < #qdev do
             local qlen=qdev[id]:size(1)
             local lowertriangle=torch.tril(torch.ones(qpool[id]:size(1),qpool[id]:size(1)))
             local per_predicted=torch.pow(lowertriangle*torch.ones(qpool[id]:size(1),1),-1) -- used to calculate reciprocal rank
              
             -- Calculate q vector
             local qmat,_=texts2matrix({qdev[id]},qlen)
             local qvec=transfer_data(torch.max(nn.JoinTable(3):forward(qvecmod:forward(qmat)),3)) 
             qvec=qvec:reshape(qvec:size(1),qvec:size(2))
             local qmatrix=transfer_data(torch.ones(qpool[id]:size(1),1))*qvec 

             for i=1,qpool[id]:size(1) do
                 amatrix[i]=avecmatrix[qpool[id][i]]
             end
             -- Cosines of all the answers in the pool
             local answer_cosines=cosinedist:forward({qmatrix,amatrix})
             local sortedvalues,sortedans= torch.sort(answer_cosines,1,true)
             local corpositions=torch.zeros(qpool[id]:size(1))
             for i=1,qcorr[id]:size(1) do
                 --print(qcorr[id][i])
                 corpositions=corpositions + sortedans:eq(i):double()
             end
             reciprocalrank[id]= torch.max(torch.diag(corpositions)*per_predicted)
             if ( reciprocalrank[id] == 1) then
                 accuracy = accuracy+1
             end
             id=id+1
    end
            

   print("-> Mean Reciprocal Rank on validation set = "..reciprocalrank:mean())
   print("-> Accuracy on validation set = "..(accuracy*100)/#qdev.."%" )
   return (accuracy*100)/#qdev 

              
end
           


local function main()

--Read from vocab file and store in table
  vocab = getvocab()
  local vocab_size=0
  for _ in pairs(vocab) do vocab_size = vocab_size + 1 end
  params.vocab_size=vocab_size
--Read and store answer dictionary
  answerdict = getanswers()
--Read and store training question dictionary
  qdict ,qtrain = get_trainingdata()
--Read and store dev file
  qdev,qpool,qcorr = get_dev()

  print("Network parameters:")
  print(params)

  print("Number of Questions:",#qdict)
  print("Number of answers:",#answerdict)
  print("###################################")

  setup()

  local iter={}
  for i=1,#qdict do
      for j=1,qtrain[i]:size(1) do
          table.insert(iter,{i,qtrain[i][j]})
       end 
  end
  
  local counter=torch.Tensor(iter)

  -- These clone modules are just to save (This will avoid saving state variables)      
  local qmod_save=qmod:clone('weight','bias')
  local maxcosmod_save=maxcosmod:clone('weight','bias')

  -- These modules are for evaluation (validation set)
  qvecmod=transfer_data(qmod:clone('weight','bias'))
  avecmod=transfer_data(qvecmod:clone('weight','bias'))

  qvecmod:evaluate()
  avecmod:evaluate()

  crit=transfer_data(nn.MarginRankingCriterion(params.margin))
  crit.sizeAverage=false

  local p1=transfer_data(torch.ones(params.batch_size))
  local zero1by1=transfer_data(torch.zeros(1))
  local one1by1=transfer_data(torch.ones(1))
  local step = 1 --5 epochs --ounter:size(1) - 20
  local epoch = 0 
  local prevepoch = 0
  local maximum_accuracy=0
  local beginning_time = torch.tic()
  --local  listofwas=getwans()
  --listofwas[12887]=torch.ones(167)*24607
  local listofwas={}


  print("#Checking validation set before training..")
  local accuracy=check_valid()
  if ( accuracy > maximum_accuracy ) then
      maximum_accuracy=accuracy
  end

  print("#Starting training..")

  local epoch_size = counter:size(1) 
  local round = 0
   

while epoch < params.max_epoch do
    

    local questions={}
    local answers={}
    local wronganswers={}
    local qlen=0
    local corra_len=0
    local wronga_len=0
    local list={} 
    local num=0

    -- Get list of wrong answers
    while num<params.batch_size do
       local id=(step-1+num)%epoch_size+1 --  this is to  loop   the counter tensor

       local qid=counter[id][1]
       local corransid=counter[id][2]
       local qmat,_=texts2matrix({qdict[qid]},qdict[qid]:size(1))
       local qvec=torch.max(nn.JoinTable(3):forward(qvecmod:forward(qmat)),3)
       qvec=qvec:reshape(qvec:size(1),qvec:size(2))


      
       if ( qlen < qdict[qid]:size(1) ) then
           qlen = qdict[qid]:size(1) --#qdict[qid]
       end
       if ( corra_len < answerdict[corransid]:size(1) ) then
           corra_len = answerdict[corransid]:size(1) --#answerdict[corransid]
       end

       -- pick most similar wrong answer from 50 randomly choosen wrong answers
       if ( listofwas[qid] == nil ) then
          local probwronganswers=torch.Tensor(params.anspool):random(1,#answerdict)
          local checkcorrs=torch.zeros(params.anspool)
          for corrid=1,qtrain[qid]:size(1) do
               checkcorrs=checkcorrs+probwronganswers:eq(qtrain[qid][corrid]):double()
          end
          -- check correct answers in the list of random answers(number of random answers=params.anspool)
          checkcorrs=(checkcorrs-1)*-1 -- compliment of 0-1 vector checkcorrs
          local a={}
          local maxlen=0
          for ans=1,params.anspool do
              table.insert(a,answerdict[probwronganswers[ans]]) 
              if ( maxlen < answerdict[probwronganswers[ans]]:size(1) ) then
                  maxlen=answerdict[probwronganswers[ans]]:size(1)
              end
           end
           local amat,_=texts2matrix(a,maxlen)
           local wavecmatrix=torch.max(nn.JoinTable(3):forward(avecmod:forward(amat)),3)
           wavecmatrix=transfer_data(wavecmatrix:reshape(wavecmatrix:size(1),wavecmatrix:size(2)))

           local qvecmatrix=transfer_data(torch.ones(params.anspool,1)*qvec) -- duplicating qvector 'params.anspool' number of times 
           
           local cosdist=nn.CosineDistance():cuda()
           local cos_qwa=cosdist:forward({qvecmatrix,wavecmatrix})
           local cosvalue,waindex=torch.max(torch.diag(checkcorrs):cuda()*cos_qwa,1)
           
           listofwas[qid] = probwronganswers[waindex[1]]
      end

      if ( wronga_len < answerdict[listofwas[qid]]:size(1) ) then
          wronga_len=answerdict[listofwas[qid]]:size(1) 
      end
      
      num=num+1


      table.insert(questions,qdict[qid])
      table.insert(answers,answerdict[corransid])
      table.insert(wronganswers,answerdict[listofwas[qid]])
     
      --print("question-answer pair ",qid,corransid,listofwas[qid],torch.toc(t))
     
    end
    
    qvecmod:clearState()
    avecmod:clearState()


    -- LOAD tensors
    local qmatrix,qmask=texts2matrix(questions,qlen)
    local amatrix,amask=texts2matrix(answers,corra_len)
    local wamatrix,wamask=texts2matrix(wronganswers,wronga_len)

    -- FORWARD PROPAGATION
    -- forget state variables in the RNN
    qmod:forget()
    amod:forget()
    wamod:forget()
    -- zerogradparameters
    qmod:zeroGradParameters()
    amod:zeroGradParameters()
    wamod:zeroGradParameters()

    local qtbl=qmod:forward(qmatrix)
    local atbl=amod:forward(amatrix)
    local watbl=wamod:forward(wamatrix)

    -- Add masked vectors to add a large negative value to the fillers' output (not necessary as we AddConstant(-100) in the module already)
    --local qvectensor=nn.JoinTable(3):cuda():forward(qtbl)+qmask
    --local avectensor=nn.JoinTable(3):cuda():forward(atbl)+amask
    --local wavectensor=nn.JoinTable(3):cuda():forward(watbl)+wamask

    local qvectensor=nn.JoinTable(3):cuda():forward(qtbl)
    local avectensor=nn.JoinTable(3):cuda():forward(atbl)
    local wavectensor=nn.JoinTable(3):cuda():forward(watbl)

    local qvectbl=qvectensor:split(1,3)
    local avectbl=avectensor:split(1,3)
    local wavectbl=wavectensor:split(1,3)

    local cosqa,cosqwa=unpack(maxcosmod:forward({qvectbl,avectbl,wavectbl}))

    cosqa=cosqa:reshape(params.batch_size,1)
    cosqwa=cosqwa:reshape(params.batch_size,1)
    local err=crit:forward({cosqa,cosqwa},p1)
    --print("err=",err)
     
    -- BACKWARD PROPAGATION
    local gradCrit=crit:backward({cosqa,cosqwa},p1)
    local gradMaxCosMod=maxcosmod:backward({qvectbl,avectbl,wavectbl},gradCrit)
    local gradqmod=qmod:backward(qmatrix,gradMaxCosMod[1]) 
    local gradamod=amod:backward(amatrix,gradMaxCosMod[2]) 
    local gradqmod=wamod:backward(wamatrix,gradMaxCosMod[3]) 
    
    --update parameters
    maxcosmod:updateParameters(params.lr)
    qmod:updateParameters(params.lr)
    amod:updateParameters(params.lr)
    wamod:updateParameters(params.lr)

    collectgarbage()

    step = step + params.batch_size
    epoch = step / epoch_size
    print("trained "..(step-1).." qa-pairs(in "..torch.toc(beginning_time).." secs)")

     
    -- Logging for every 1/10th epoch
    if (epoch - prevepoch) >= 0.1  then
           print("trained "..epoch.." epochs in "..torch.toc(beginning_time).." secs")
          torch.save('qamod.net',qmod_save)
          torch.save('maxcosmod.net',maxcosmod_save)
          print("->Saved nets amod.net,qamod.net ")
          listofwas={} -- referesh this table for later epochs
          prevepoch=epoch
    end

   if ( step >  round*epoch_size  ) then 
      round=round+1

      -- decreasing lr
      --params.lr=1/(round+14)
      --print(" Changing lr to ".. params.lr)

     local accuracy=check_valid()
     local accuracy=0
     if   ( accuracy > maximum_accuracy ) then
        maximum_accuracy=accuracy
        print("->Maximum accuracy so far="..maximum_accuracy)
        torch.save('qamod.best.net',qvecmod)
    end
   end
    
  end


 print("Training is over.")

end

main()
