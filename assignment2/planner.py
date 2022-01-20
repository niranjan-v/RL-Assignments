import argparse
import numpy as np
import random
import pulp
from pulp import solvers
##
def read_mdp(mdp):
	lines = [line.rstrip('\n') for line in open(mdp)]
	line0=lines[0].split()
	line1=lines[1].split()
	num_states=int(line0[0])
	num_actions=int(line1[0])
	##
	R=np.zeros([num_states, num_actions, num_states], dtype='double')
	T=np.zeros([num_states, num_actions, num_states], dtype='double')
	##
	base=2
	for i in range(base,base+num_states*num_actions):
		state_num=(i-base)//num_actions
		action_num=(i-base)%num_actions
		line=lines[i].split()
		rews=np.array([float(x) for x in line])
		R[state_num, action_num]=rews

	base=2+num_states*num_actions
	for i in range(base,base+num_states*num_actions):
		state_num=(i-base)//num_actions
		action_num=(i-base)%num_actions
		line=lines[i].split()
		tprobs=np.array([float(x) for x in line])
		T[state_num, action_num]=tprobs

	base=2+num_states*num_actions*2

	line0=lines[base].split()
	line1=lines[base+1].split()

	gamma=float(line0[0])
	ftype=line1[0]

	return num_states, num_actions, R, T, gamma, ftype

##
def lp(mdp):
	S, A, R, T, gamma, ftype=read_mdp(mdp)
	prob = pulp.LpProblem('mdpsolver', pulp.LpMinimize)
	if ftype=='episodic':
		num_var=S-1
	else:
		num_var=S
	dvars=[pulp.LpVariable('v{}'.format(i), cat='Continuous') for i in range(num_var)]
	total_cost = ""
	for i in range(len(dvars)):
		total_cost += dvars[i]
	prob+=total_cost
	##constraints
	for s in range(S):
		if ftype=='episodic' and s==S-1:
			continue
		for a in range(A):
			bnd=0.0
			expr=""
			for s1 in range(S):
				bnd-=T[s,a,s1]*R[s,a,s1]
				if ftype=='episodic' and s1==S-1:
					continue
				if s==s1:
					expr+=(gamma*T[s,a,s1]-1.0)*dvars[s1]
				else:
					expr+=gamma*T[s,a,s1]*dvars[s1]
			prob+= (expr <= bnd)
	optimization_result = prob.solve()#solvers.PULP_CBC_CMD(fracGap=0.000000000001)
	V=np.zeros(S,dtype='double')
	try:
		for i in range(len(dvars)):
			V[i]=dvars[i].varValue+0.0
	except:
		print("problem is infeasible!")
		exit(1)
	#print(V)
	P=[]
	for s in range(S):
		Q=np.zeros(A,dtype='double')
		for a in range(A):
			for s1 in range(S):
				Q[a]+=T[s,a,s1]*(R[s,a,s1]+gamma*V[s1])
		P.append(np.argmax(Q))
	
	for s in range(S):
		print(np.around(V[s],decimals=10), P[s])#

####

def policy_eval(S, A, R, T, gamma, P, ftype):
	prob = pulp.LpProblem('mdpsolver', pulp.LpMinimize)
	if ftype=='episodic':
		num_var=S-1
	else:
		num_var=S
	dvars=[pulp.LpVariable('v{}'.format(i), cat='Continuous') for i in range(num_var)]
	for s in range(S):
		if ftype=='episodic' and s==S-1:
			continue
		expr=""
		bnd=0.0
		a=P[s]
		for s1 in range(S):
			bnd-=T[s,a,s1]*R[s,a,s1]
			if ftype=='episodic' and s1==S-1:
				continue
			if s==s1:
				expr+=(gamma*T[s,a,s1]-1.0)*dvars[s1]
			else:
				expr+=gamma*T[s,a,s1]*dvars[s1]
		prob+= (expr == bnd)
	##
	optimization_result = prob.solve()#solvers.PULP_CBC_CMD(fracGap=0.000000000001)
	V=np.zeros(S,dtype='double')
	try:
		for i in range(len(dvars)):
			V[i]=dvars[i].varValue+0.0
	except:
		print("problem is infeasible!")
		exit(1)
	return V

def hpi(mdp):
	S, A, R, T, gamma, ftype=read_mdp(mdp)
	P=np.random.randint(A, size=S)
	converged=False
	while not converged:
		converged=True
		V=policy_eval(S, A, R, T, gamma, P, ftype)
		###improvement
		for s in range(S):
			Q=np.zeros(A,dtype='double')
			for a in range(A):
				for s1 in range(S):
					Q[a]+=T[s,a,s1]*(R[s,a,s1]+gamma*V[s1])
			if not P[s]==np.argmax(Q):
				converged=False
				P[s]=np.argmax(Q)

	for s in range(S):
		print(np.around(V[s],decimals=10), P[s])#

##main
parser = argparse.ArgumentParser()
parser.add_argument('--mdp', action='store',  required=True)
parser.add_argument('--algorithm', action='store', default="lp") #lp, hpi
args = parser.parse_args()
mdp=args.mdp
alg=args.algorithm

globals()[alg](mdp)
