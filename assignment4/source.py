import numpy as np
import os, sys
import matplotlib.pyplot as plt
import argparse
'''
SETUP: same as example 6.5 sutton & barto
7*10 grid
upward wind strengths : [0 0 0 1 1 1 2 2 1 0]
'''

R=7
C=10
winds=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
start=30
end=37

mov=np.array([[-1,0],[0,1],[1,0],[0,-1],[-1,1],[1,1],[1,-1],[-1,-1]])


def bnd(x,y):
	return max(min(x,R-1),0), max(min(y,C-1),0)

def next_state(s,a,p=1.):	
	# directions: x-cur position
	# 7 0 4 
	# 3 x 1
	# 6	2 5
	x=s//C
	y=s%C
	ws=winds[y]
	if p<1:
		ws=np.random.choice([ws-1,ws,ws+1],p=[(1-p)/2,p,(1-p)/2])
	x1=x+mov[a,0]-ws
	y1=y+mov[a,1]
	x1,y1=bnd(x1,y1)
	return x1*C+y1

def eps_soft_init(S,A,eps):
	T=np.ones((S,A))*(eps/(1.0*A))
	x=np.arange(S)
	y=np.random.randint(A, size=S)
	T[x,y]+=1.0-eps
	return T

def sarsa(nS,nA,num_eps=170,alpha=0.5,epsilon=0.1,p=1.):
	Q=np.zeros((nS,nA))
	T=eps_soft_init(nS,nA,epsilon)
	epnum={}
	for eps in range(num_eps):
		S=start
		A=np.random.choice(np.arange(nA), p=T[S])
		while not S==end:
			if eps in epnum:
				epnum[eps]+=1
			else:
				epnum[eps]=1
			S1=next_state(S,A,p)
			R=-1
			A1=np.random.choice(np.arange(nA), p=T[S1])
			Q[S,A]+=alpha*(R+Q[S1,A1]-Q[S,A])
			##update T
			Aopt=np.argmax(Q[S])
			T[S]=np.ones(nA,)*(epsilon/(1.0*nA))
			T[S,Aopt]+=1.0-epsilon
			##
			S,A=S1,A1
	return epnum

def update_cnts(epnums,i):
	for x in epnums:
		if x in cnts[i]:
			cnts[i][x]+=epnums[x]
		else:
			cnts[i][x]=epnums[x]

def getl(epnums):
	res=[]
	for x in epnums:
		res+=[x]*epnums[x]
	return res

###

parser = argparse.ArgumentParser()
parser.add_argument('--stats', action='store',  type=int, default=0) 
parser.add_argument('--exp', action='store',  type=int, default=3) 
args = parser.parse_args()
rs=int(args.stats)
num_exp=int(args.exp)
###
cnts=[{} for _ in range(num_exp)]
base_dir="plots/"

os.makedirs(base_dir, exist_ok=True)
for s in range(10):
	##
	np.random.seed(s)
	##
	stitle="seed {}".format(s)
	for i in range(num_exp):
		if i==0:
			nA=4
			title="4 actions"
			p=1.
		elif i==1:
			nA=8
			title="8 actions"
			p=1.
		elif i==2:
			nA=8
			p=1.0/3.0
			title="8 actions stochastic p={0:.2f}".format(p)
		elif i==3:
			nA=8
			p=2.0/3.0
			title="8 actions stochastic p={0:.2f}".format(p)
		elif i==4:
			nA=4
			p=1.0/3.0
			title="4 actions stochastic p={0:.2f}".format(p)
		elif i==5:
			nA=4
			p=2.0/3.0
			title="4 actions stochastic p={0:.2f}".format(p)
			
		epnum=sarsa(70,nA,p=p)
		fig=plt.figure(1)
		lst=getl(epnum)
		update_cnts(epnum,i)
		plt.plot(np.arange(len(lst)), lst, label=title)
	fig.suptitle(stitle, fontsize=20)
	plt.xlabel('Time steps',fontsize=16)
	plt.ylabel('Episodes',fontsize=16)
	plt.legend(loc='best')
	plt.savefig(base_dir+'seed_{}.png'.format(s))
	plt.close(fig)
	#plt.show()

for i in range(num_exp):
	for x in cnts[i]:
		cnts[i][x]=cnts[i][x]//10

##avg plot
finres=[]
for i in range(num_exp):
	if i==0:
		title="4 actions"	
	elif i==1:
		title="8 actions"
	elif i==2:
		p=1.0/3.0
		title="8 actions stochastic p={0:.2f}".format(p)
	elif i==3:
		p=2.0/3.0
		title="8 actions stochastic p={0:.2f}".format(p)
	elif i==4:
		p=1.0/3.0
		title="4 actions stochastic p={0:.2f}".format(p)
	elif i==5:
		p=2.0/3.0
		title="4 actions stochastic p={0:.2f}".format(p)

	fig=plt.figure(1)
	lst=getl(cnts[i])
	finres.append(len(lst))
	plt.plot(np.arange(len(lst)), lst, label=title)
fig.suptitle("average plot", fontsize=20)
plt.xlabel('Time steps',fontsize=16)
plt.ylabel('Episodes',fontsize=16)
plt.legend(loc='best')
plt.savefig(base_dir+'avg.png')
plt.close(fig)

if rs>0:
	print("Exp num\tNumber of steps")
	for x in range(len(finres)):
		print("{}\t{}".format(x, finres[x]))

