import argparse
import numpy as np
import random

#####
def kldiv(p,q):
	# if q==0.0 or q==1.0:
	# 	print("error")
	return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def getucb(p,n,k):
	thresh=1e-10
	lb=p+thresh
	ub=1-thresh
	if lb>=ub:
		return p
	if p<1e-9:
		p=1e-9
	ct=k/n #(np.log(t)+3*np.log(np.log(t)))/n
	for _ in range(24):
		if (-lb+ub)<=1e-7:
			break
		c_m=ct-kldiv(p,(lb+ub)/2)
		if c_m<=0:
			ub=(lb+ub)/2
		else:
			lb=(lb+ub)/2
	return lb
	


######
def roundrobin(ins, al, rs, eps, hz):
	marr=np.loadtxt(ins)
	#print(marr)
	lt=len(marr)
	rew=0
	np.random.seed(rs)
	samples=np.random.uniform(0,1,hz)
	for i in range(hz):
		cm=marr[i%lt] #curr mean
		if samples[i]<=cm:
			rew+=1
	reg=max(marr)*hz-rew
	print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,hz,reg))
	#instance, algorithm, random seed, epsilon, horizon, REG 

######
def epsilongreedy(ins, al, rs, eps, hz):
	marr=np.loadtxt(ins)
	lt=len(marr)
	rew=0
	np.random.seed(rs)
	samples=np.random.uniform(0,1,hz*2)
	##
	rt=np.zeros(lt, dtype=int) # num of rew 
	cts=np.zeros(lt, dtype=int) # num of times arm is pulled
	qt=np.zeros(lt, dtype=float) #estimated action values=rt./cts
	##
	cidx=0
	for i in range(hz):
		##
		if samples[2*i]<=eps: #explore
			cidx=np.random.randint(lt)
		else:
			cidx=np.argmax(qt)
		##
		cts[cidx]+=1
		if samples[2*i+1]<=marr[cidx]: #rew=1
			rew+=1
			rt[cidx]+=1			
		qt[cidx]=rt[cidx]/cts[cidx]

	reg=max(marr)*hz-rew
	print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,hz,reg))


######
def ucb(ins, al, rs, eps, hz):
	marr=np.loadtxt(ins)
	lt=len(marr)
	rew=0
	np.random.seed(rs)
	samples=np.random.uniform(0,1,hz)
	#
	if hz<=lt:
		for i in range(hz):
			cm=marr[i]
			if samples[i]<cm:
				rew+=1
		reg=max(marr)*hz-rew
		print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,hz,reg))
		return
	##
	rt=np.zeros(lt, dtype=int) # num of rew 
	cts=np.ones(lt, dtype=int) # num of times arm is pulled
	##
	for i in range(lt):
		cm=marr[i]
		if samples[i]<cm:
			rew+=1
			rt[i]+=1
	qt=rt/cts
	#cidx=argmax(ucbt)
	for i in range(lt,hz):
		ucbt=qt+np.sqrt(2*np.log(i)/cts)
		cidx=np.argmax(ucbt)
		cts[cidx]+=1
		if samples[i]<=marr[cidx]:
			rew+=1
			rt[cidx]+=1
		qt[cidx]=rt[cidx]/cts[cidx]

	reg=max(marr)*hz-rew
	print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,hz,reg))

######
def klucb(ins, al, rs, eps, hz):
	marr=np.loadtxt(ins)
	lt=len(marr)
	rew=0
	np.random.seed(rs)
	samples=np.random.uniform(0,1,hz)
	#
	if hz<=lt:
		for i in range(hz):
			cm=marr[i]
			if samples[i]<cm:
				rew+=1
		reg=max(marr)*hz-rew
		print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,hz,reg))
		return
	##
	rt=np.zeros(lt, dtype=int) # num of rew 
	cts=np.ones(lt, dtype=int) # num of times arm is pulled
	##
	for i in range(lt):
		cm=marr[i]
		if samples[i]<=cm:
			rew+=1
			rt[i]+=1
	qt=rt/cts
	#qt.astype(double)
	#cidx=argmax(ucbt)
	for i in range(lt,hz):
		####remove later
		# if i in [50,200,800,3200,12800,51200]:
		# 	reg=max(marr)*i-rew
		# 	print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,i,reg))
		####
		k=(np.log(i)+3*np.log(np.log(i)))
		ucbt=np.array([getucb(qt[j],cts[j],k) for j in range(lt)] )
		cidx=np.argmax(ucbt)
		cts[cidx]+=1
		if samples[i]<=marr[cidx]:
			rew+=1
			rt[cidx]+=1
		qt[cidx]=rt[cidx]/cts[cidx]

	reg=max(marr)*hz-rew
	print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,hz,reg))

######
def thompsonsampling(ins, al, rs, eps, hz):
	marr=np.loadtxt(ins)
	lt=len(marr)
	rew=0
	np.random.seed(rs)
	samples=np.random.uniform(0,1,hz)
	#
	st=np.zeros(lt,dtype=int)
	ft=np.zeros(lt,dtype=int)
	#
	rew=0
	for i in range(hz):
		arms_arr=np.random.beta(1+st,1+ft) #np.array([np.random.beta(1+st[j],1+ft[j]) for j in range(lt)])
		cidx=np.argmax(arms_arr)
		if samples[i]<=marr[cidx]:
			st[cidx]+=1
			rew+=1
		else:
			ft[cidx]+=1

	reg=max(marr)*hz-rew
	print("{}, {}, {}, {}, {}, {}".format(ins,al,rs,eps,hz,reg))


#if __name__ == "__main__":
###args parsing
#TODO: take care of partial args
parser = argparse.ArgumentParser()

parser.add_argument('--instance', action='store',  required=True)
parser.add_argument('--algorithm', action='store', default="round-robin")
parser.add_argument('--randomSeed', action='store',  type=int, default=1) #, required=True
parser.add_argument('--epsilon', action='store', type=float, default=0.5) 
parser.add_argument('--horizon', action='store',  type=int, default=50)

args = parser.parse_args()

ins=args.instance
al=args.algorithm
rs=int(args.randomSeed)
eps=float(args.epsilon)
hz=int(args.horizon)

random.seed()
fn=al.replace("-", "")
globals()[fn](ins, al, rs, eps, hz)
# for rs in range(50):
# 	globals()[fn](ins, al, rs, eps, hz)