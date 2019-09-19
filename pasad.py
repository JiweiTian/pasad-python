import numpy as np
from numpy import linalg
from scipy.linalg import hankel
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

arguments=sys.argv[2:]
count=len(arguments)
r=1

def PASAD_TRAIN(train,n,l):
	X=hankel(train[0:l],train[l:n])
	u,s,vt = linalg.svd(X)
	U = u[:,:r]
	return U.T


data=genfromtxt(sys.argv[1], delimiter=",")
train_1=data[:500,:]
train_2=data[500:4000,:]
test=data[4000:4800,:]
	
L=250
N1=train_1.shape[0]
N2=train_2.shape[0]
K=N2-L+1
S=train_1.shape[1]

for i in arguments:
	i=int(i)
	print("Starting " + str(i) + " Sensor")
	UT = PASAD_TRAIN(train_1[:,i-1],N1,L)
	print("Pasad Complete")					#First Learning Parameter (UT)
	s=np.zeros(L)
	s=s.T
	for j in range(0,K-1):
		H=hankel(train_2[0:L,i-1],train_2[L:N2,i-1])
		H.reshape((250,H.shape[1]))
		xi=H[:,j]
		#print(xi.shape)	
		s=s+xi
		#print(s.shape)

	c=s/K
	cc=UT.dot(c)							#Second Learning Parameter (cc)
	print("Centroid found")	
	d=np.zeros(K)
	
	for j in range(0,K-1):	
		d[j]=np.linalg.norm(cc-UT.dot(H[:,j]))
	
	dmax=np.amax(d)							#  Third Learning Parameter (dmax) 
	print("Departure Calculated: " + str(dmax))
	N=N1+N2
	x=train_2[N2-L:N2,i-1]
	Tsize=test.shape[0]
	D=np.zeros(Tsize)
	for j in range(N+1,4800):
		#print(x.shape)
		x=x[1:]	
		#print(x.shape)
		x=np.insert(x,L-2,test[j-N,i-1])
		x=x.T
		y= cc-UT.dot(x)
		D[j-N-1]=(y.T).dot(y)
		
	Dmax=np.amax(D)
		
	if (Dmax> dmax):
		print("Hola ! Hola ! Hola ! Alarm triggered! Dmax: " +str(Dmax))
	else:
		print("No Alarm! Dmax: "+str(Dmax))



	dd=np.insert(d,K,D)
	
	plt.subplot(2,1,1)
	plt.ylabel('XMEAS('+str(i)+')')
	plt.plot(np.arange(0,4801,1),data[:,i-1])

	plt.subplot(2,1,2)
	plt.xlabel('Time')
	plt.ylabel('Departure Score')
	plt.plot(np.arange(750,4801,1),dd)
	plt.plot(np.arange(750,4801,1),np.repeat(dmax,4051))
	prefix=sys.argv[1]		
	plt.savefig(prefix[5:8]+'_sensor_'+str(i)+'.pdf')
	plt.close()
			
	






