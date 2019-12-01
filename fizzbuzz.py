import numpy as np
import torch
import torch.nn as nn

def fizz_buzz_encode(i):
    if i%15==0:return 3
    elif i%5==0:return 2
    elif i%3==0:return 1
    else:return 0
def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]
def helper(i):
    print(fizz_buzz_decode(i,fizz_buzz_encode(i)))

#for i in range(16):
 #   helper(i)

NUM_DIGITS=10

#把数字转成二进制数字，特征多一点
def binary_encode(i,num_digits):
    return np.array(([i>>d &1 for d in range(num_digits)]))

trX=torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(101,2**NUM_DIGITS)])
trY=torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGITS)])
NUM_HIDDEN=100

model=torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS,NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN,4)
)
if torch.cuda.is_available():
    model.cuda()
#分类问题用CrossEntropy
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)
BATCH_SIZE=128

for epoch in range(10000):
    for start in range(0,len(trX),BATCH_SIZE):
        end=start+BATCH_SIZE
        batchX=trX[start:end]
        batchY=trY[start:end]
        if torch.cuda.is_available():
            batchX=batchX.cuda()
            batchY=batchY.cuda()
        y_pred=model(batchX)
        loss=loss_fn(y_pred,batchY)
        #print("epoch",epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

textX=torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(1,101)])
if torch.cuda.is_available():
    textX=textX.cuda()
with torch.no_grad():
    testY=model(textX)
prediction=zip(range(1,101),testY.max(1)[1].cpu().data.tolist())
#预测的也太准了
print([fizz_buzz_decode(i,x) for i,x in prediction])