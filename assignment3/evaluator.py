import numpy as np
import sys


def evaluate(fpath):
	def read_path(fpath):
		lines = [line.rstrip('\n') for line in open(fpath)]
		lines = [line.split() for line in lines]
		###
		S=int(lines[0][0])
		A=int(lines[1][0])
		gamma=float(lines[2][0])
		###
		seq=[int(lines[i][0]) for i in range(3,len(lines))]
		rew=[float(lines[i][2]) for i in range(3,len(lines)-1)]
		###
		return rew,seq,gamma,S

	rew,seq,gamma,S=read_path(fpath)
	##
	V=np.zeros(S, dtype='double')
	etrace=np.zeros(S, dtype='double')
	alpha=0.95
	lmbd=0.89
	epochs=np.min([S,25])
	for ep in range(epochs):
		for i in range(len(seq)-1):
			ti=ep*len(seq)+i
			# alpha= 0.3/((ti+1)**(0.6)) #0.30/((ti+1)**(0.6)) 0.0131
			# if alpha<0.025:
			# 	alpha=0.025
			s=seq[i]
			ns=seq[i+1]
			r=rew[i]
			##
			delta=r+gamma*V[ns]-V[s]
			etrace[s]=1.0
			# print(delta, etrace)
			V=V+alpha*delta*etrace
			etrace=gamma*lmbd*etrace

			if ti%650==0 and alpha > 70e-6:
				alpha*=0.492


	##
	for i in range(len(V)):
		print(V[i])

###
##vanilla method for deterministic policies

def evaluate_lin(fpath):
	def get_mdp(fpath):
		lines = [line.rstrip('\n') for line in open(fpath)]
		lines = [line.split() for line in lines]
		S=int(lines[0][0])
		A=int(lines[1][0])
		gamma=float(lines[2][0])
		###
		R=np.zeros([S, S], dtype='double')
		Tc=np.zeros([S, S], dtype='double')
		###
		
		for i in range(3,len(lines)-1):
			s=int(lines[i][0])
			r=float(lines[i][2])
			ns=int(lines[i+1][0])
			##
			R[s,ns]=r
			Tc[s,ns]+=1.0
			##
		Ts=Tc.sum(axis=1)
		T=Tc/Ts[:,np.newaxis]
		#smoothing####
		df=1e-3
		T=T*(1-df)+df/S
		###############
		return R,T,gamma,S
	##
	R,T,gamma,S=get_mdp(fpath)
	# print(R,T,gamma)
	##
	A=np.zeros([S, S], dtype='double')
	B=np.zeros(S, dtype='double')
	for i in range(S):
		A[i][i]-=1
		for j in range(S):
			B[i]-=T[i,j]*R[i,j]
			A[i,j]+=T[i,j]*gamma
	##
	V=np.linalg.solve(A,B)
	##
	for i in range(len(V)):
		print(V[i])


###
if not len(sys.argv)==2:
	print("invalid args")
	exit(1)

evaluate(sys.argv[1])
