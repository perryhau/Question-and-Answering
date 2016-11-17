-----------------------------------------------------------------------------------
-- Name:        read_qacopus.lua
-- Purpose:     To read 'insuranceQA' like corpus(https://github.com/shuzi/insuranceQA.git)
-- Written by:  Stalin Varanasi
-- Remarks:     -Change the global variables to suit your file locations.
--              -Write your own read_qacorpus.lua to fit corpora of other formats
-----------------------------------------------------------------------------------

require 'cutorch/init'
require 'cunn/init'


-- Global variables 
vocabfile="./data/vocabulary"
answersfile="./data/answers.txt"
trainingfile="./data/question.train.txt"
validationfile="./data/question.dev.txt"
listofwasfile="./data/list_of_wronganswers.txt"
word2vecfile="./data/Words2Vectors"


function table.contains(table, element)
  for _, value in pairs(table) do
    if value == element then
      return true
    end
  end
  return false
end


function transfer_data(x)
  return x:cuda()
end




-- To get the list of all the wrong answers prior to training 
function getwans()

   local file=io.open('list_of_wronganswers.txt','r')
   local all=file:read("*all")
   local wronganswers={}
   for i=1,12887 do
       wronganswers[i]={}
   end
   for line in string.gmatch(all,"([^\n]*)\n") do
          local idend,wordbegin = string.find(line,'%s+',1)
          local q = string.sub(line,1,idend-1)
          local wa = string.sub(line,wordbegin+1)
          if ( not table.contains(wronganswers[tonumber(q)],tonumber(wa) ) ) then 
                 table.insert(wronganswers[tonumber(q)],tonumber(wa))
          end
   end


   for i=1,12887 do
       wronganswers[i]=torch.Tensor(wronganswers[i])
   end

return wronganswers

end


function getvocab()
  local vocabfile,msg = io.open(vocabfile,"r")
  local all = vocabfile:read("*all")
  vocab = {}

   for line in string.gmatch(all,"([^\n]*)\n") do
        local idend,wordbegin = string.find(line,'%s+',1)
        local id = string.sub(line,1,idend-1)
        local word = string.sub(line,wordbegin+1)
        vocab[id]=word
   end
   return vocab
end


function getanswers()
 local ansdictfile,msg = io.open(answersfile,"r")
  local all = ansdictfile:read("*all")
  answerdict = {}
   for line in string.gmatch(all,"([^\n]*)\n") do
        local idend,ansbegin = string.find(line,'%s+',1)
        local id = string.sub(line,1,idend-1)
        local ans = string.sub(line,ansbegin+1)
        local anstbl={}
        for token in string.gmatch(ans, "[^%s]+") do
            local id,pos= string.gsub(token,"idx_","")
            table.insert(anstbl,tonumber(id))
        end
        --table.insert(answerdict,anstbl)
        answerdict[tonumber(id)]=transfer_data(torch.Tensor(anstbl))


   end
   return answerdict
end

function get_trainingdata()
      -- Read and store training question dictionary
  local qdictfile,msg = io.open(trainingfile,"r")
  local all = qdictfile:read("*all")
  qdict = {}
  qtrain={}
  local qid=1

  for line in string.gmatch(all,"([^\n]*)\n") do
        local qtbl={} -- table of words (question.)
        local qtr={}  -- table of answers to be put in training table
        for token in string.gmatch(line, "[^%s]+") do
             if string.match(token,"^idx") then
                local id,pos=string.gsub(token,"^idx_","")
                table.insert(qtbl,tonumber(id))
             else
                table.insert(qtr,token)
             end
        end
        qdict[qid]=transfer_data(torch.Tensor(qtbl))
        qtrain[qid]=torch.Tensor(qtr)
        qid = qid+1
   end
   return qdict,qtrain
end



function get_dev()
  local qdictfile,msg = io.open(validationfile,"r")
  local all = qdictfile:read("*all")
  qdev = {}
  qpool={}
  qcorr={}
   local qid=1

   for line in string.gmatch(all,"([^\n]*)\n") do
        local qtbl={}
        local atbl={}
        local cortbl={}
        local qstart=0
        for token in string.gmatch(line, "[^%s]+") do
             if string.match(token,"^idx") then
                local id,pos=string.gsub(token,"^idx_","")
                table.insert(qtbl,tonumber(id))
                qstart=1
             else
                if ( qstart == 1) then
                    local id,pos=string.gsub(token,"^idx_","")
                    table.insert(atbl,id)
                else
                    local id,pos=string.gsub(token,"^idx_","")
                    table.insert(cortbl,id)
                end
             end
         end 
        qdev[qid]=transfer_data(torch.Tensor(qtbl))
        qpool[qid]=transfer_data(torch.Tensor(atbl))
        qcorr[qid]=transfer_data(torch.Tensor(cortbl))
        qid = qid+1
   end
   return qdev,qpool,qcorr
end


