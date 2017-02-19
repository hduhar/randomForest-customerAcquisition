#cython: boundscheck=False,wraparound=False
from libcpp.stack cimport stack
import numpy as np
from libc.math cimport log2,sqrt,log,exp,pow
from libc.stdlib cimport malloc, free
import sys
from libcpp.map cimport map
from libcpp.vector cimport vector
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libcpp.algorithm cimport sort as stdsort
from cython.parallel import parallel, prange,threadid
from libcpp.map cimport pair
import matplotlib.pyplot as plt
from cython.operator cimport dereference as deref, preincrement as inc


#make cython know about time.h in c
cdef extern from "time.h":
	struct tm:
		int tm_sec
		int tm_mday
		int tm_mon
		int tm_year
	tm* localtime(long *timer) nogil
	long time(long *tloc) nogil

cdef struct data_struct:
	long** left
	long** right

#useful composite struct
cdef struct Four:
	double first
	double second
	double third
	double fourth
	long fifth
	#long *sixth
	map[long, long] sixth

#useful composite struct
cdef struct Five:
	double first
	double second
	long third
	long fourth
	long fifth
	#long* sixth
	map[long, long] sixth


#define the node of tree
#here isLeaf is set when the node is leaf
#here isCategorical is set when the splitting attribute is categorical
#left and right are pointers to left and right child
#proba is probability of being in class 0
#availAtt is list of available attributes on which the node can be split
cdef struct Node:
	long height,attribute,isCategorical,availableAttLen,isLeaf
	double attributeValue,infoGain,proba,entropy
	Node *left
	Node *right
	long *availableAtt


#Record that is pushed into the stack
#data not contained in the node so that it can be freed easily
cdef struct StackRecord:
	long dataLen
	Node *tree_node
	long** idx_data


#return the entropy for two probabilities x and y
#entropy of 0log0 is taken 0
cdef double entropy(double x,double y) nogil:
	##print prob_tup[0],prob_tup[1]
	cdef double kk
	if x==0 or y==0:
		return 0.0
	kk=-x*log2(x)-y*log2(y)
	return kk


#allocates and return pointer to 2D array 
#containing r rows and c columns
cdef long** allocate(long r,long c) nogil:
	cdef long i
	cdef long **arr = <long **>malloc(r * sizeof(long *))
	for i in range(r):
		arr[i] = <long *>malloc(c * sizeof(long))
	return arr

#frees the memory allocated to a 2D array
#takes in the pointer, row and column of the array
cdef void free_mem(long **arr,long r,long c) nogil:
	cdef long i
	for i in range(r):
		free(arr[i])
	free(arr)
	return


#calculates the probability of a list of ids of samples
#return the probability of positive and negative in a set of samples
cdef Five calProb(long* id_data,long dataLen) nogil:
	#here data is just an array of ids

	cdef double countp=0.0
	cdef double countn=0.0
	cdef Five result
	cdef long i
	##print len(data)
	#if dataLen is zero then probabiliies are assumed to be zero
	if dataLen==0:
		result.first=0.0
		result.second=0.0
		return result

	#else calculate the nos of positive and negative by iterating over the data
	for i in range(dataLen):
		if data[id_data[i],0]==1:
			countp+=1.0
		else:
			countn+=1.0
	#reult.third=countp	
	result.first=countp/dataLen
	result.second=countn/dataLen

	return result



#calculates the information gain when the attribute is continuous
#checks for split only when there is change in class
cdef Four calContInfo(long* id_data,long att,long e,double currEntropy,long dataLen) nogil:
	##here the data is sorted list of ids 
	##defines two classes leftclass=-1 and rightclass=1
	cdef double infoGain=-1
	cdef double value=data[id_data[0],e]
	#cdef double value=data[0][0][e]
	cdef double leftNeg=0.0
	cdef double leftPos=0.0
	cdef double rightNeg=0.0
	cdef double rightPos=0.0
	cdef double now_class=data[id_data[0],0]
	cdef double skipPos=0.0
	cdef double skipNeg=0.0
	cdef double trueProb, falseProb, trueEntropy, falseEntropy, iGain, expectedEntropy
	cdef long i,leftPoints
	cdef Four result
	cdef long split_index
	#cdef long* classified=<long *>malloc(sizeof(long)*dataLen)
	cdef map[long,long] classified
	for i in range(dataLen):
		#temp[i]=now_class
		classified[id_data[i]]=1
		if data[id_data[i],0]==0.0:
			rightNeg+=1.0
	rightPos=dataLen-rightNeg
	split_index=0

	for i in range(1,dataLen):
		if data[id_data[i],0]==now_class:
			#temp[i]=now_class
			if now_class==1.0:
				skipPos+=1.0
			else:
				skipNeg+=1.0
		else:
			if now_class==1.0:
				leftPos+=skipPos+1.0
				rightPos-=skipPos+1.0
				skipPos=0.0
			else:
				leftNeg+=skipNeg+1.0
				rightNeg-=skipNeg+1.0
				skipNeg=0.0
			now_class=(now_class+1.0)%2
			#< is true and >= is false
			#information gain =currEntropy- expected entropy
			trueProb=(leftNeg+leftPos)/(<double>dataLen)
			falseProb=(rightNeg+rightPos)/(<double>dataLen)
			
			trueEntropy=entropy(leftPos/(leftPos+leftNeg),leftNeg/(leftPos+leftNeg))
			falseEntropy=entropy(rightPos/(rightPos+rightNeg),rightNeg/(rightPos+rightNeg))
			expectedEntropy=trueProb*trueEntropy+falseProb*falseEntropy
			iGain=currEntropy-expectedEntropy
			if infoGain<iGain:
				infoGain=iGain
				#value=data[i][0][e]
				split_index=i
				value=data[id_data[i],e]
				leftPoints=<long>(leftNeg+leftPos)
				##print (leftPoints)
	
	for i in range(split_index):
		classified[id_data[i]]=-1
	result.first=infoGain
	result.second=value
	result.fifth=leftPoints
	result.sixth=classified
	return result



#calculates the information gain when the attribute is categorical 
#categorical attributes have been assumed to be binary
cdef Four calCatInfo(long* id_data,long att,long e,double currEntropy,long dataLen) nogil:
	##here the data is sorted list of ids 
	cdef long i
	cdef long *trueList=<long *>malloc(sizeof(long)*dataLen)
	cdef long *falseList=<long *>malloc(sizeof(long)*dataLen)
	cdef long true_ind=0
	cdef long false_ind=0
	cdef double x,y
	cdef Five result
	cdef double trueEntropy,falseEntropy,trueProb,falseProb,expectedEntropy,infoGain
	cdef Four ret_result
	cdef map[long,long] classified

	for i in range(dataLen):
		if data[id_data[i],e]==1:
			trueList[true_ind]=id_data[i]
			true_ind+=1
			classified[id_data[i]]=-1
		else:
			falseList[false_ind]=id_data[i]
			false_ind+=1
			classified[id_data[i]]=1
	
	result=calProb(trueList,true_ind)
	x=result.first
	y=result.second
	trueEntropy=entropy(x,y)

	result=calProb(falseList,false_ind)
	x=result.first
	y=result.second
	falseEntropy=entropy(x,y)
	
	trueProb=(<double>true_ind)/dataLen
	falseProb=(<double>false_ind)/dataLen
	expectedEntropy=trueProb*trueEntropy+falseProb*falseEntropy
	infoGain=currEntropy-expectedEntropy
	
	ret_result.first=infoGain
	ret_result.second=-99
	ret_result.fifth=true_ind#leftpoints
	free(trueList)
	free(falseList)
	#return [infoGain,trueList,falseList]
	#return [infoGain,None]
	#nothing assigned to map check
	ret_result.sixth=classified
	return ret_result



#reads data from csv into numpy array 
#feature_isCat is set when that particular feature is categorical
def readData(f):
	'''
	Reads data from file into numpy
	#'has_bought_brand_category': 8-3,
	#'has_bought_brand_company': 7-3,
	#'has_bought_brand_company_category': 6-3,
	# 'never_bought_brand': 5-3,
	# 'never_bought_category': 4-3,
	# 'never_bought_company': 3-3,

	'''
	data = np.genfromtxt(f, dtype=float, delimiter=' ', skip_header=1)
	data=np.delete(data,1,1)#remove repeattrip label
	cdef long num_features,dataLen
	##delete id and offer id also
	data=np.asfortranarray(data,dtype=float)#change to column major
	##print 'oho'+str(data.shape)
	
	yy=np.delete(data,1,1)#remove id
	#yy=np.delete(yy,1,1)#remove offer id #offer_id not available now
	yy=np.delete(yy,0,1)#remove labels
	yy=np.argsort(yy,axis=0)
	cdef long[:,:] data_idx_sorted = np.asfortranarray(yy,dtype=int)#labels deleted
	num_features=data_idx_sorted.shape[1]
	dataLen=data_idx_sorted.shape[0]
	#need to assign cat features
	cdef long[::1] feature_isCat=np.zeros(num_features,dtype=int)
	feature_isCat[0]=1
	feature_isCat[1]=1
	feature_isCat[2]=1
	feature_isCat[3]=1
	feature_isCat[4]=1
	feature_isCat[5]=1
	
	for i in range(56,93):
		feature_isCat[i]=1

	##print data_idx_sorted


	return data,data_idx_sorted,feature_isCat,num_features,dataLen



#return the best attribute to spliit on
#checks for each attribute the most information gain
cdef Five bestFeature(long** idx_data,Five probTup,long *avail_attri,long avail_attriLen,long dataLen) nogil:
	##here the data is list of list 
	cdef double max_info_gain=-1
	cdef double value
	cdef long attribute
	cdef double currEntropy=entropy(probTup.first,probTup.second)
	cdef long e,att,data_ind,flag,leftPoints,isCategorical
	cdef Four a
	cdef Five b
	cdef long r
	cdef map[long,long] classified_map
	cdef long tttime
	cdef long chosen_index
	cdef long max_features_num=<long>log2(avail_attriLen)
	cdef long *chosen_att=<long *>malloc(sizeof(long)*max_features_num)

	tttime=time(NULL)
	srand(tttime)
	
	for r in range(max_features_num):
		chosen_index=rand()%avail_attriLen
		chosen_att[r]=avail_attri[chosen_index]

	##can be taken in func defn
	#cdef double dataLen=len(data[0])
	##print avail_attriLen
	for e in range(max_features_num):
		##print "<"+str(e)
		att=chosen_att[e]
		#data_ind=att+3#e=index_dict[att]
		data_ind=att+2
		if feature_isCat[att]==1:
			a=calCatInfo(idx_data[0],att,data_ind,currEntropy,dataLen)
			flag=1
		else:
			#list_indd=list_order_map[att]
			a=calContInfo(idx_data[att],att,data_ind,currEntropy,dataLen)
			flag=0
		if a.first>=max_info_gain:
			max_info_gain=a.first
			if flag==1:
				isCategorical=1
				#value=None
				attribute=att
				leftPoints=a.fifth
				classified_map=a.sixth
			else:
				isCategorical=0
				value=a.second
				b.second=value
				attribute=att
				leftPoints=a.fifth
				classified_map=a.sixth
				
		
	#sys.exit()
	b.first=max_info_gain
	b.third=isCategorical
	b.fourth=attribute
	b.fifth=leftPoints
	b.sixth=classified_map
	#return [max_info_gain,value,isCategorical,attribute,leftPoints]
	return b








#builds decision tree using depth first technique
#min_sample_split is nos of data samples required to create a split
cdef Node* buildDecisionTree(long** ids_data,long dataLen,long max_height,long *avail_attri,long num_features,long min_sample_split) nogil:
	##here the data is numpy array as fotran array
	##delete data to be added
	##print "Enter buildDecisionTree"
	cdef Node *root=<Node *>malloc(sizeof(Node)*1)

	cdef stack[StackRecord *] mystack
	cdef StackRecord *curr
	#cdef long iter1,iter2
	cdef Five info
	cdef pair[long,double] entry

	cdef long *attList1
	cdef long *attList2
	cdef long** leftList
	cdef long** rightList
	cdef Five probTup
	cdef long attt,att_index,iter_temp
	cdef long *temp
	cdef data_struct *split_data

	cdef Node *leftChild
	cdef Node *rightChild
	
	cdef StackRecord *leftSR
	cdef StackRecord *rightSR
	cdef map[long,double].iterator it
	


	root.height=1
	root.availableAtt=avail_attri #need to apply more logic for decision tree
	root.availableAttLen=num_features
	
	
	curr=<StackRecord *>malloc(sizeof(StackRecord))
	curr.tree_node=root
	
	curr.dataLen=dataLen
	curr.idx_data=ids_data
	mystack.push(curr)
	curr=NULL
	
	while(not mystack.empty()):
		curr=<StackRecord *>mystack.top()
		mystack.pop()
		probTup=calProb(curr.idx_data[0],curr.dataLen)
		if probTup.first==0.0:
			curr.tree_node.proba=1.0
			curr.tree_node.isLeaf=1
		elif probTup.second==0.0:
			curr.tree_node.proba=0.0
			curr.tree_node.isLeaf=1
		elif curr.tree_node.availableAttLen==0:
			curr.tree_node.proba=probTup.second
			curr.tree_node.isLeaf=1
		elif curr.dataLen<min_sample_split:
			curr.tree_node.proba=probTup.second
			curr.tree_node.isLeaf=1
		elif curr.tree_node.height==max_height:
			curr.tree_node.proba=probTup.second
			curr.tree_node.isLeaf=1
		else:
			#[max_info_gain,value,isCategorical,attribute,leftPoints]
			info=bestFeature(curr.idx_data,probTup,curr.tree_node.availableAtt,curr.tree_node.availableAttLen,curr.dataLen)
			curr.tree_node.infoGain=info.first
			curr.tree_node.isCategorical=info.third
			curr.tree_node.attributeValue=info.second
			curr.tree_node.attribute=info.fourth
			if curr.tree_node.infoGain==0.0:
				curr.tree_node.proba=probTup.second
				curr.tree_node.isLeaf=1
			else:
				if curr.tree_node.isCategorical==1:
					att_index=0
				
					attList1=<long *>malloc((curr.tree_node.availableAttLen-1)*sizeof(long))
					attList2=<long *>malloc((curr.tree_node.availableAttLen-1)*sizeof(long))
					for attt in range(curr.tree_node.availableAttLen):
						if curr.tree_node.availableAtt[attt]!=curr.tree_node.attribute:
							attList1[att_index]=curr.tree_node.availableAtt[attt]
							attList2[att_index]=curr.tree_node.availableAtt[attt]
							att_index+=1
				else:
					att_index=0
				
					attList1=<long *>malloc((curr.tree_node.availableAttLen)*sizeof(long))
					attList2=<long *>malloc((curr.tree_node.availableAttLen)*sizeof(long))
					for attt in range(curr.tree_node.availableAttLen):
						attList1[att_index]=curr.tree_node.availableAtt[attt]
						attList2[att_index]=curr.tree_node.availableAtt[attt]
						att_index+=1
					
				##loop
				#info.fifth is leftpoints
				if curr.tree_node.isCategorical==1:
					split_data=divCatDataset(curr.idx_data,curr.tree_node.attribute,info.fifth,curr.dataLen,num_features,info.sixth)#leftList,rightList)
				else:
					split_data=divContDataset(curr.idx_data,curr.tree_node.attribute,curr.tree_node.attributeValue,info.fifth,curr.dataLen,num_features,info.sixth)
				leftList=split_data.left
				rightList=split_data.right
				
				leftSR=<StackRecord*>malloc(sizeof(StackRecord))
				rightSR=<StackRecord*>malloc(sizeof(StackRecord))
				#print (curr.dataLen,info.fifth,curr.dataLen-info.fifth)
				
				leftChild=<Node *>malloc(sizeof(Node))
				rightChild=<Node *>malloc(sizeof(Node))

				leftSR.idx_data=leftList
				rightSR.idx_data=rightList
				leftSR.dataLen=info.fifth
				rightSR.dataLen=curr.dataLen-info.fifth
				#print curr.dataLen,info.fifth,rightSR.dataLen

				leftChild.height=curr.tree_node.height+1
				rightChild.height=curr.tree_node.height+1


				#testing here insert map here
				##comment this section when graph is not needed
				##making graph slows down the process
				'''

				with gil:				
					it=score_map.find(curr.tree_node.attribute)
					if it!=score_map.end():
						entry.first=deref(it).first
						entry.second=deref(it).second+log(<double>exp(10/<double>pow(curr.tree_node.height,8)))
						score_map.erase(it)
						score_map.insert(entry)
					else:
						entry.first=curr.tree_node.attribute
						entry.second=log(<double>exp(10/<double>pow(curr.tree_node.height,8)))
						score_map.insert(entry)
					


				'''
				
				
				leftChild.availableAtt=attList1
				rightChild.availableAtt=attList2
				leftChild.availableAttLen=curr.tree_node.availableAttLen-1
				rightChild.availableAttLen=curr.tree_node.availableAttLen-1
				curr.tree_node.isLeaf=0
				curr.tree_node.left=leftChild
				curr.tree_node.right=rightChild
				
				leftSR.tree_node=leftChild
				rightSR.tree_node=rightChild
				
				mystack.push(rightSR)
				mystack.push(leftSR)
				rightSR=NULL
				leftSR=NULL
				leftChild=NULL
				rightChild=NULL
				attList1=NULL
				attList2=NULL

		free_mem(curr.idx_data,num_features,curr.dataLen)
		free(curr)
		curr=NULL
	return root

cdef data_struct* divCatDataset(long** idx_data,long attribute,long leftPoints,long dataLen,long n_features,map[long,long] classified) nogil:
	cdef long e=attribute+2
	cdef data_struct *split
	cdef long i,j,new_i,new_j
	split=<data_struct *>malloc(sizeof(data_struct))
	split.left=allocate(n_features,leftPoints)
	split.right=allocate(n_features,dataLen-leftPoints)
	#new_j=0
	#new_i=0
	for i in range(n_features):
		new_j=0
		new_i=0
		for j in range(dataLen):

			if classified[idx_data[i][j]]==-1:
				split.left[i][new_j]=idx_data[i][j]
				new_j+=1
			else:
				split.right[i][new_i]=idx_data[i][j]
				new_i+=1
		if new_j+new_i!=dataLen:
			pass
			#print "problem-hee"+str(new_j+new_i)+"<>"+str(dataLen)
			#with gil:
			#	#print "problem here"+str(new_j+new_i)+"<>"+str(dataLen)
	return split



cdef data_struct* divContDataset(long** idx_data,long attribute,double attributeValue,long leftPoints,long dataLen,long n_features,map[long,long] classified) nogil:
	cdef long e=attribute+2
	cdef data_struct *split
	cdef long i,j,new_i,new_j
	split=<data_struct *>malloc(sizeof(data_struct))
	split.left=allocate(n_features,leftPoints)
	split.right=allocate(n_features,dataLen-leftPoints)
	##print data_idx_sorted
	count=0
	for i in range(n_features):
		new_j=0
		new_i=0

		for j in range(dataLen):
			##print (new_j,new_i)
			if classified[idx_data[i][j]]==-1:
				split.left[i][new_j]=idx_data[i][j]
				new_j+=1
			else:
				split.right[i][new_i]=idx_data[i][j]
				new_i+=1
		if new_j+new_i!=dataLen:
			pass

	return split
	

cdef double classify_prob(double[::1] dataPoint,Node *root,long isProb):
	#Here dataPoint is a list of features
	#returns only prob of being 1
	cdef Node *curr=root
	while(True):
		if curr.isLeaf==1:
			if isProb==1:
				return 1-curr.proba
			else:
				if curr.proba>=0.5:
					return 0.0
				else:
					return 1.0
		else:
			#index=curr.attribute+2
			index=curr.attribute+1
			if curr.isCategorical==1:
				if dataPoint[index]==1:
					curr=curr.left
				else:
					curr=curr.right
			else:
				if dataPoint[index]<curr.attributeValue:
					curr=curr.left
				else:
					curr=curr.right


cdef double* classify_data(long isProb,Node *root):
	#return a list of class/probs in order
	#if isProb is True return a list of tuple containing (proba,1-proba)
	#cdef double *testProb=<double *>malloc(sizeof(double)*testLen)
	cdef long i
	cdef double probb
	cdef double *probabs=<double *>malloc(sizeof(double)*testLen)
	for i in range(testLen):
		probb=classify_prob(testSet[i,:],root,isProb)
		probabs[i]=probb
	return probabs
	

def readTest(f):
	#cdef long dataLen
	#cdef double[:,::1] lol
	data = np.genfromtxt(f, dtype=float, delimiter=' ', skip_header=1)
	data=np.delete(data,0,1)#delete label
	data=np.delete(data,0,1)#delete
	dataLen=data.shape[0]
	#data=data.astype(np.float32)
	#for e,row in enumerate(data):
	#	testSet.append(list(row))
	return data,dataLen


cdef double* build_bag_tree(map[long,long] sample_map,long sampleSize,long dataLen,long height,long *features,long n_features) nogil:
	cdef long i
	cdef long k
	cdef long j,tttemp
	cdef long index,count
	cdef long** sampleData
	cdef double* probs
	cdef Node *tree_root=NULL
	#with gil:
	#	print (n_features,sampleSize)
	sampleData=allocate(n_features,sampleSize)
	count=0
	p=0
	for i in range(n_features):
		index=0
		for k in range(dataLen):
			if sample_map.find(data_idx_sorted[k,i])!=sample_map.end():
				tttemp=sample_map[data_idx_sorted[k,i]]
				for j in range(tttemp):
					sampleData[i][index]=data_idx_sorted[k,i]
					index+=1
				#with gil:
				#	#pass
				#	print (int(sampleData[i][index]),threadid())


	tree_root=buildDecisionTree(sampleData,sampleSize,height,features,n_features,50)
	#free_mem(sampleData,n_features,sampleSize)
	with gil:
		probs=classify_data(1,tree_root)
	#with gil:
	#	print ("completed")
	return probs


cdef double* bag_tree(long sampleSize,long dataLen,long height,long *features,long n_features) nogil:
	cdef long k
	#int *a=<int *>malloc(sizeof(int)*sampleSize)
	cdef map[long,long] sample_map
	cdef long ttemp=0
	cdef long tttime
	#with gil:
	tttime=time(NULL)
	srand(tttime)
	#sample_map[-1]=-1
	for k in range(sampleSize):
		ttemp=rand()%dataLen
		if sample_map.find(ttemp)!=sample_map.end():
			sample_map[ttemp]=sample_map[ttemp]+1
		else:
			sample_map[ttemp]=1
	return build_bag_tree(sample_map,sampleSize,dataLen,height,features,n_features)


cdef double* bagging_trees(long sampleSize,long num_trees,long dataLen,long height,long *features,long n_features,long this_testLen,long nthreads):
	cdef long i
	cdef long k
	cdef long j
	cdef double* probs
	cdef double *mean_local
	cdef double* mean_probs=<double *>malloc(sizeof(double)*this_testLen)
	
	for i in range(this_testLen):
		mean_probs[i]=0.0
	
	with nogil, parallel(num_threads=nthreads):
		
		mean_local=<double *>malloc(this_testLen*sizeof(double))

		for k in range(this_testLen):
			mean_local[k]=0.0
		
		for i in prange(num_trees):
			probs=bag_tree(sampleSize,dataLen,height,features,n_features)
			
			for j in range(this_testLen):
				mean_local[j]+=probs[j]
			free(probs)
			probs=NULL
		
		with gil:
			for k in range(this_testLen):
				mean_probs[k]+=mean_local[k]
		
		free(mean_local)
		mean_local=NULL
	
	for k in range(this_testLen):
		mean_probs[k]=mean_probs[k]/num_trees
	
	return mean_probs
		






##########################################################
# Main of the program
##########################################################
train='./train/train_hot_encoded.csv'
test='./test/test_hot_encoded.csv'
cdef double[::1,:] data
cdef long[::1,:] data_idx_sorted
cdef long[::1] feature_isCat
cdef long num_features,dataLen
cdef double* probs
#cdef long testLen
#cdef double[:,::1] testSet
#cdef long testLen

cdef map[long,double] *score_map=new map[long,double]()
cdef map[long,double].iterator it
cdef Node *tree_root
data,data_idx_sorted,feature_isCat,num_features,dataLen=readData(train)
##print data_idx_sorted
##print data_idx_sorted.ndim
cdef long *feature_i=<long *>malloc(sizeof(long)*num_features)

for k in range(num_features):
	feature_i[k]=k
testSet,testLen=readTest(test)
probs=bagging_trees(dataLen,100,dataLen,15,feature_i,num_features,testLen,4)



outfile = open('./results/random_forest_probs', "wb")
outfile.write('id,repeatProbability\n')
for k in range(testLen):
	outfile.write(str(long(testSet[k,0]))+','+str(probs[k])+'\n')
outfile.close()


##plotting variable importance graph
##comment this when graph is not needed
'''
X=[]
Y=[]
it=score_map.begin()
while it!=score_map.end():
	X.append(deref(it).first)
	Y.append(deref(it).second)
	inc(it)
for i in range(len(X)):
	print (X[i],Y[i])
width=0.5
plt.bar(X,Y,width,color="blue",tick_label=X)
plt.savefig('variable_importance_random_forest.png')
#plt.xticks(X,X)

plt.show()
'''
