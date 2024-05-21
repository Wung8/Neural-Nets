import sys; args = sys.argv[1:]
import math, time, random, re
import csv
import numpy as np
import pickle
from scipy.special import expit

start = time.process_time()

training_file = 'mnist_train.csv'


# MATRIX FUNCTIONS

def matDims(mat1):
  return (len(mat1),len(mat1[0]))

def dotProdct(vec1,vec2):
  return sum(a*b for a,b in zip(vec1,vec2))

def matMult(mat1,mat2):
  return np.matmul(mat1,mat2)
  if matDims(mat1)[1] != matDims(mat2)[0]: print('invalid')
  tr = [[0] for j in range(len(mat1))]
  for a in range(len(mat1)):
    vec1 = mat1[a]
    vec2 = [mat2[b][0] for b in range(len(mat2))]
    tr[a][0] = dotProdct(vec1,vec2)
  return tr

def rowMult(mat1,vec1):
  return [[mat1[j][i]*vec1[j] for i in range(len(mat1[0]))] for j in range(len(mat1))]

def matAdd(mat1,mat2):
  return [[mat1[j][i]+mat2[j][i] for i in range(len(mat1[0]))] for j in range(len(mat1))]


def rotateMat(mat1):
  return [[mat1[i][j] for i in range(len(mat1))] for j in range(len(mat1[0]))]

def copyMat(mat1):
  return [mat1[i][:] for i in range(len(mat1))]


# ACTIVATION FUNCTIONS

def T1_linear(vec1):
  return vec1

def T2_ramp(vec1):
  return [[max(x[0],0)] for x in vec1]

def T3_logistic(vec1):
  return [[expit(x[0])] for x in vec1]

def T4_logistic(vec1):
  return [[2/(1+math.e**-x[0])-1] for x in vec1]

ACTFUNCDCT = {'T1':T1_linear,
           'T2':T2_ramp,
           'T3':T3_logistic,
           'T4':T4_logistic
          }

def reverseSigmoid(i):
  if i == 1: return 99999
  return math.log(i/(1-i))

def sigmoid(i):
  if i > 20: return 0.99999
  if i < -20: return 0.00001
  return 1/(1+math.e**-i)

def sigmoidDeriv(i):
  return sigmoid(i)*(1-sigmoid(i))

def deriv(i):
  return i*(1-i)

# NETWORK FUNCTIONS

def feedForward(inpt):
  global VALUES
  x = inpt
  VALUES = [x.copy()]
  for i in range(len(WEIGHTS)):
    #x = np.matmul(WEIGHTS[i],x)
    x = matMult(WEIGHTS[i],x)
    x = np.array(ACTIVATION(x))
    VALUES.append(x.copy())
  #x = [[b[0]*FINALWEIGHTS[a][0]] for a,b in enumerate(x)]
  return x

def quickForward(inpt):
  x = inpt
  for i in range(len(WEIGHTS)):
    #x = np.matmul(WEIGHTS[i],x)
    x = matMult(WEIGHTS[i],x)
    x = ACTIVATION(x)
  return x

K = .1
P = 0

def backPropogate_Linear(error):
  global K
  errors = [[0 for i in range(j)] for j in LAYERLENS]
  errors[-1] = [[i] for i in error]
  for i in range(len(errors)-2,-1,-1):
    #errors[i] = np.matmul(rotateMat(WEIGHTS[i]),errors[i+1])
    errors[i] = matMult(rotateMat(WEIGHTS[i]),errors[i+1])
  for i in range(len(WEIGHTS)):
    gradient = copyMat(WEIGHTS[i])
    values = VALUES[i]
    error = errors[i+1]
    for j in range(len(gradient)):
      for k in range(len(gradient[0])):
        gradient[j][k] *= -K*(abs(x:=error[j][0])**.1)*[-1,1][x<0]*values[k][0]
    WEIGHTS[i] = matAdd(WEIGHTS[i],gradient)    


def crossEntropyLoss(a,y): return -(y*math.log(a)+(1-y)*math.log(1-a))

def crossEntropyLossDeriv(a,y): return (y/a+(1-y)/(1-a))

def error(y,t):
  return sum([(y[i][0]-t[i])**2 for i in range(len(y))])

def checkGradient(g,i):
  global x,y
  epsilon = 10**-7

  a = random.randint(0,len(WEIGHTS[i])-1)
  b = random.randint(0,len(WEIGHTS[i][0])-1)

  WEIGHTS[i][a][b] += epsilon
  plus = error(quickForward(x),y)
  WEIGHTS[i][a][b] -= 2*epsilon
  minus = error(quickForward(x),y)
  d = (plus-minus)/(2*epsilon)
  WEIGHTS[i][a][b] += epsilon

  

  #print(str(d)[:7],str(g[a][b])[:7])
  print(str(d),'   ',str(g[a][b]),'   ',d+g[a][b])

  
  
   
def backPropogate_Logistic(error):
  global K,P,gradient
  #K = .5
  P = 0
  values = VALUES[-1]
  errors = [np.zeros(j) for j in LAYERLENS]
  errors[-1] = error
  for i in range(len(errors)-2,-1,-1):
    values = VALUES[i]
    errors[i] = np.matmul(rotateMat(WEIGHTS[i]),errors[i+1])
    for j in range(len(errors[i])):
      errors[i][j][0] *= deriv(values[j][0])

  for i in range(len(WEIGHTS)):
    gradient = [[0 for a in range(len(WEIGHTS[i][b]))] for b in range(len(WEIGHTS[i]))]
    values = VALUES[i]
    nextvalues = VALUES[i+1]
    error = errors[i+1]
    for j in range(len(WEIGHTS[i])):
      for k in range(len(WEIGHTS[i][j])):
        gradient[j][k] = K*error[j][0]*values[k][0]

    #if random.random()<1/len(WEIGHTS[i]): checkGradient(gradient,i)
    
    WEIGHTS[i] = matAdd(WEIGHTS[i],gradient)

 

def initWeights(a,b):
  weights = [[random.random()*2-1 for i in range(a)] for j in range(b)]
  WEIGHTS.append(weights)

def setGradient():
  global gradient
  gradient = []
  for i in range(len(WEIGHTS)):
    gradient.append([[0 for a in range(len(WEIGHTS[i][b]))] for b in range(len(WEIGHTS[i]))])

def applyGradient():
  global WEIGHTS,gradient,batchsize
  for i in range(len(WEIGHTS)):
    WEIGHTS[i] = matAdd(WEIGHTS[i],gradient[i])

print('reading file')

data = []
with open("mnist_train.csv",'r') as csvfile:
  csvreader = csv.reader(csvfile, delimiter = ',')
  check = 6000
  for c,row in enumerate(csvreader):
    if c==0: continue
    if c==6001: break
    if c%check==0: print(str(c//check*10),'%')
    x = [[int(i)/255] for i in row[1::]]+[[1]]
    #y = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
    #y[int(row[0])] = [1]
    y = [0,0,0,0,0,0,0,0,0,0]
    y[int(row[0])] = 1
    x,y = np.array(x),np.array(y)
    data.append([x,y])

print('finished reading file')
   
inputlen = len(data[0][0])
outputlen = len(data[0][1])

print(inputlen)

#print('DATA :',data)


ACTIVATION = ACTFUNCDCT['T3']

LAYERLENS = [inputlen,300,10,outputlen]
#LAYERLENS = [inputlen,100,10,1]

WEIGHTS = []
for a,b in zip(LAYERLENS,LAYERLENS[1:]):
  initWeights(a,b)


with open(str(0)+'MNIST.pkl', 'rb') as f:
  WEIGHTS = pickle.load(f)

print('loaded weights')


print('initialized weights')

def getMI(lst):
  return lst.index(max(lst))


HIST = 0
MIN = 999
def redonplateu(e):
  global K,HIST,MIN
  if e==0: return
  if HIST>=20:
    K /= 10
    HIST = 0
    print('REDUCING LEARNING RATE TO',K)
  if e < MIN:
    HIST = 0
    MIN = e
    print('NEW MIN :D')
  HIST += 1
  
  

P2 = 0
def main():
  global start,batchsize,x,y
  start = time.time()
  K = .00001
  batchsize = 10
  batch = -1
  c = 0
  check = 50
  avge = 0
  accuracy = 0
  print('started training')
  a = 0
  oldavge,oldaccuracy = 0,0
  for xxx in range(len(data)//batchsize):
    if xxx%10 == 0: print(str(time.time()-start)[:10],(int(xxx/(len(data)//batchsize)*100)),'%')
    #print(int(xxx/(len(data)//batchsize)*100))
    batch += batchsize
    #setGradient()
    print('|',end='')
    for x,y in data[batch:batch+batchsize]:
      #print(y)
      c += 1
      if c%6000==0:
        print('SAVING...')
        with open(str(c//6000)+'MNIST.pkl', 'wb') as f:
          pickle.dump(WEIGHTS,f)
      output = feedForward(x)
      error = np.array([[y[i]-output[i][0]] for i in range(outputlen)])
      #error = [(v:=y[i]-output[i][0])*abs(v) for i in range(outputlen)]
      if P2 and (c%check==0): print(y[0],str(output[0][0])[:5],str(error[0]/abs(v))[:7])
      backPropogate_Logistic(error)
      if np.argmax(output)==np.argmax(y): accuracy += 1
      avge += sum(abs(i) for i in error)/10
      a += 1
      if a==100:
        print(str(avge/a)[:5],str(accuracy),'|',str(sum(abs(i[0]) for i in error))[:5],np.argmax(output),np.argmax(y),[str(ii[0])[:5] for ii in output])
        avge,accuracy,a = 0,0,0
        oldavge,oldaccuracy = 0,0
          #f.write('\n'.join([str([[str(w)[:5] for w in weightlayer] for weightlayer in weights]) for weights in WEIGHTS]))
    print('   ',avge-oldavge,accuracy-oldaccuracy)
    #redonplateu(avge-oldavge)
    oldavge,oldaccuracy = avge,accuracy
    #applyGradient()

main()

#main()

##Michael Wang, 2024, Pd7
