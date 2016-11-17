


-- qa pattern1 :
-- The qa_pattern1_file should contain questions and answers in this pattern
-- #question 
-- <questin text>
-- #answer
-- <answer text>
---------------------



if ( #arg ==  0 ) then
   print("Usage: th makeitlikeins.lua <qa_pattern1_file>")
   os.exit()
else
   qa_pattern1_file=arg[1]
end

file,msg=io.open(qa_pattern1_file,"r")
qfile,msg=io.open("./question.train.txt","w")
dfile,msg=io.open("./question.dev.txt","w")
tfile,msg=io.open("./question.test.txt","w")
afile,msg=io.open("./answers.txt","w")
vfile,msg=io.open("./vocabulary","w")

all=file:read("*all")
anspool=50


questions={}
answers={}
vocab={}
rvocab={}

count={}

for line in string.gmatch(all,"([^\n]+)\n") do
    for word in string.gmatch(line,"([^\t ]+)[\t ]") do
       if ( count[word] == nil ) then
            count[word]=1
       else
            count[word]=count[word]+1

       end
    end
end


idx=1

local qflag
local qno=0
for line in string.gmatch(all,"([^\n]+)\n") do

  if ( line == "#question" ) then
       qflag=true
       qno=qno+1
  elseif ( line == "#answer" ) then
       qflag=false
  elseif ( not string.match(line,"^[\t ]*$")) then
    local newline=" "
    for word in string.gmatch(line,"([^\t ]+)[\t ]") do
       if ( count[word] == 1 ) then
            word="#oov"
       end
       if ( vocab[word] == nil ) then
            vocab[word]=idx
            rvocab[idx]=word
            idx=idx+1
       end
       newline=newline.." idx_"..vocab[word]
       --newline=newline.." "..word
    end

    if ( qflag ) then
        table.insert(questions,newline.."\t"..qno)
        --qfile:write(newline.."\t"..qno.."\n")
        --print(newline.."\t"..qno.."\n")
    else
        table.insert(answers,qno.."\t"..newline)
        afile:write(qno.."\t"..newline.."\n")
    end
  end
end


for i=1,#rvocab do
     vfile:write("idx_"..i.." "..rvocab[i].."\n")
end   


traincount=0.8 * qno
devcount=0.1 * qno
testcount=0.1 * qno

print("#total","#train","#dev","#test")
print(qno,traincount,devcount,testcount)

j=1
for i,v in pairs(questions) do
   if ( j <= traincount ) then
      qfile:write(v.."\n")
   elseif( j<= traincount+devcount) then
      q,no=unpack(v:split("\t"))
      local extra=torch.Tensor(anspool):random(1,qno)
      while ( extra:eq(no):sum() == 1 ) do
            extra=torch.Tensor(anspool):random(1,qno)
      end 
      extra[1]=no
      str=""
      for el=1,extra:size(1) do
         str=str.." "..extra[el]
      end
      dfile:write(no.."\t"..q.."\t"..str.."\n")
   else
      q,no=unpack(v:split("\t"))
      local extra=torch.Tensor(anspool):random(1,qno)
      while ( extra:eq(no):sum() == 1 ) do
            extra=torch.Tensor(anspool):random(1,qno)
      end 
      extra[1]=no
      str=""
      for el=1,extra:size(1) do
         str=str.." "..extra[el]
      end
      tfile:write(no.."\t"..q.."\t"..str.."\n")
   end
   j=j+1
end

