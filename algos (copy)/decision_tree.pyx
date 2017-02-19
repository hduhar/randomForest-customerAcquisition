#cython: boundscheck=False
from libcpp.stack cimport stack
import numpy as np
from libc.math cimport log2
from libc.stdlib cimport malloc, free
import sys
#from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
import time



###This file takes into account the offer_id as one_hot_encoded
cdef struct data_struct:
	long** left
	long** right

cdef struct Four:
	double first
	double second
	double third
	double fourth
	long fifth
	#long *sixth
	map[long, long] sixth

cdef struct Five:
	double first
	double second
	long third
	long fourth
	long fifth
	#long* sixth
	map[long, long] sixth

cdef struct Node:
	long height,attribute,isCategorical,availableAttLen,isLeaf
	double attributeValue,infoGain,proba,entropy
	Node *left
	Node *right
	long *availableAtt

cdef struct StackRecord:
	long dataLen
	Node *tree_node
	long** idx_data


cdef double entropy(double x,double y) nogil:
	##print prob_tup[0],prob_tup[1]
	cdef double kk
	if x==0 or y==0:
		return 0.0
	kk=-x*log2(x)-y*log2(y)
	return kk

cdef long** allocate(long r,long c) nogil:
	#cdef long a[row][col]
	#return a
	##print r,c
	cdef long i
	cdef long **arr = <long **>malloc(r * sizeof(long *))
	for i in range(r):
		arr[i] = <long *>malloc(c * sizeof(long))
	return arr

cdef void free_mem(long **arr,long r,long c):
	cdef long i
	for i in range(r):
		free(arr[i])
	free(arr)
	return

cdef Five calProb(long* id_data,long dataLen) nogil:
	#here data is just an array of ids
	#cdef long chk_len =sizeof(<void *>id_data)/sizeof(long)
	##print "voila "+str(chk_len)

	cdef double countp=0.0
	cdef double countn=0.0
	cdef Five result
	cdef long i
	##print len(data)
	if dataLen==0:
		result.first=0.0
		result.second=0.0
		return result

	for i in range(dataLen):
		if data[id_data[i],0]==1:
			countp+=1.0
		else:
			countn+=1.0
	#reult.third=countp	
	result.first=countp/dataLen
	result.second=countn/dataLen

	return result

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
	#with gil:
		#print ("Cal_Continfo")
		#print (leftNeg,leftPos,rightNeg,rightPos,infoGain,leftPoints)

	return result



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
	#with gil:
		#print ("Cal_CATtinfo")
		#print (infoGain,true_ind)

	return ret_result


def readData(f):
	'''
	Reads data from file into numpy
	Returns a list of list of tuples
	list_order--.get the 
	tuple--.(point,label)
	point---.list of features
	
	#'has_bought_brand_category': 8-3,
	#'has_bought_brand_company': 7-3,
	#'has_bought_brand_company_category': 6-3,
	# 'never_bought_brand': 5-3,
	# 'never_bought_category': 4-3,
	# 'never_bought_company': 3-3,

	'''
	#data = np.genfromtxt(f, dtype=float, delimiter=' ', skip_header=1)
	data=np.loadtxt(f, dtype=float, delimiter=' ',skiprows=1)
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


cdef Five bestFeature(long** idx_data,Five probTup,long *avail_attri,long avail_attriLen,long dataLen) nogil:
	##here the data is list of list 
	cdef double max_info_gain=-1
	cdef double value
	cdef long attribute
	cdef double currEntropy=entropy(probTup.first,probTup.second)
	cdef long e,att,data_ind,flag,leftPoints,isCategorical
	cdef Four a
	cdef Five b
	cdef map[long,long] classified_map
	##can be taken in func defn
	#cdef double dataLen=len(data[0])
	##print avail_attriLen
	for e in range(avail_attriLen):
		##print "<"+str(e)
		att=avail_attri[e]
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








cdef Node* buildDecisionTree(long[::1,:] ids_data,long dataLen,long max_height,long *avail_attri,long num_features) nogil:
	##here the data is numpy array as fotran array
	##delete data to be added
	##print "Enter buildDecisionTree"
	cdef Node *root=<Node *>malloc(sizeof(Node)*1)

	cdef stack[StackRecord *] mystack
	cdef StackRecord *curr
	cdef long iter1,iter2
	cdef Five info

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


	root.height=1
	root.availableAtt=avail_attri #need to apply more logic for decision tree
	root.availableAttLen=num_features
	
	
	curr=<StackRecord *>malloc(sizeof(StackRecord))
	curr.tree_node=root
	
	curr.dataLen=dataLen
	curr.idx_data=allocate(num_features,curr.dataLen)#ids_data
	for iter1 in range(num_features):
		for iter2 in range(curr.dataLen):
			curr.idx_data[iter1][iter2]=ids_data[iter2,iter1]
			
	##print "<" + str(iter1)+"<"+str(iter2)+">"

	mystack.push(curr)
	curr=NULL
	
	#print "Start While"
	while(not mystack.empty()):
		with gil:
			print "hello while"
		curr=<StackRecord *>mystack.top()
		mystack.pop()
		probTup=calProb(curr.idx_data[0],curr.dataLen)
		if probTup.first==0.0:
			#print "hello1"
			curr.tree_node.proba=1.0
			curr.tree_node.isLeaf=1
		elif probTup.second==0.0:
			#print "hello2"
			curr.tree_node.proba=0.0
			curr.tree_node.isLeaf=1
		elif curr.tree_node.availableAttLen==0:
			#print "hello3"
			curr.tree_node.proba=probTup.second
			curr.tree_node.isLeaf=1
		elif curr.tree_node.height==max_height:
			#print "hello4"
			curr.tree_node.proba=probTup.second
			curr.tree_node.isLeaf=1
		else:
			#[max_info_gain,value,isCategorical,attribute,leftPoints]
			#print "hello5"
			info=bestFeature(curr.idx_data,probTup,curr.tree_node.availableAtt,curr.tree_node.availableAttLen,curr.dataLen)
			curr.tree_node.infoGain=info.first
			curr.tree_node.isCategorical=info.third
			curr.tree_node.attributeValue=info.second
			curr.tree_node.attribute=info.fourth
			with gil:
				print curr.tree_node.attribute
			if curr.tree_node.infoGain==0.0:
				#print "hello6"
				curr.tree_node.proba=probTup.second
				curr.tree_node.isLeaf=1
			else:
				#print "hello7"
				att_index=0
				
				attList1=<long *>malloc((curr.tree_node.availableAttLen-1)*sizeof(long))
				attList2=<long *>malloc((curr.tree_node.availableAttLen-1)*sizeof(long))
				for attt in range(curr.tree_node.availableAttLen):
					if curr.tree_node.availableAtt[attt]!=curr.tree_node.attribute:
						attList1[att_index]=curr.tree_node.availableAtt[attt]
						attList2[att_index]=curr.tree_node.availableAtt[attt]
						att_index+=1

				##loop
				#info.fifth is leftpoints
				#split_data=<data_struct *>malloc(sizeof(data_struct))
				if curr.tree_node.isCategorical==1:
					#print "hello8"
					split_data=divCatDataset(curr.idx_data,curr.tree_node.attribute,info.fifth,curr.dataLen,num_features,info.sixth)#leftList,rightList)
					#break
					#print "hello9"
				else:
					#print "hello10"
					split_data=divContDataset(curr.idx_data,curr.tree_node.attribute,curr.tree_node.attributeValue,info.fifth,curr.dataLen,num_features,info.sixth)
					#print "hello11"
				leftList=split_data.left
				rightList=split_data.right
				
				leftSR=<StackRecord*>malloc(sizeof(StackRecord))
				rightSR=<StackRecord*>malloc(sizeof(StackRecord))
				#print (curr.dataLen,info.fifth,curr.dataLen-info.fifth)
				
				leftChild=<Node *>malloc(sizeof(Node))
				rightChild=<Node *>malloc(sizeof(Node))
				#print "hello12"

				leftSR.idx_data=leftList
				rightSR.idx_data=rightList
				leftSR.dataLen=info.fifth
				rightSR.dataLen=curr.dataLen-info.fifth
				#print curr.dataLen,info.fifth,rightSR.dataLen
				with gil:
					print (curr.tree_node.attribute,curr.tree_node.isCategorical,curr.tree_node.attributeValue)#fo.fifth,curr.dataLen-info.fifth)		
				leftChild.height=curr.tree_node.height+1
				rightChild.height=curr.tree_node.height+1
				
				#print "hello13"

				leftChild.availableAtt=attList1
				rightChild.availableAtt=attList2
				leftChild.availableAttLen=curr.tree_node.availableAttLen-1
				rightChild.availableAttLen=curr.tree_node.availableAttLen-1
				#print "hello14"
				curr.tree_node.isLeaf=0
				curr.tree_node.left=leftChild
				curr.tree_node.right=rightChild
				
				leftSR.tree_node=leftChild
				rightSR.tree_node=rightChild
				
				#print "hello15"
				mystack.push(rightSR)
				mystack.push(leftSR)
				rightSR=NULL
				leftSR=NULL
				leftChild=NULL
				rightChild=NULL
				attList1=NULL
				attList2=NULL

				#free_mem(leftList,num_features,info.fifth)
				#free_mem(rightList,num_features,curr.dataLen-info.fifth)
				#leftList=NULL
				#rightList=NULL
				#print "hello16"
		#free_mem(curr.idx_data,num_features,curr.dataLen)
		free(curr)
		curr=NULL
		#free(temp)
		#curr=NULL
		#temp=NULL
		#attList1=NULL
		#attList2=NULL
		#del curr
	#gc.collect()
	#print "hello final"
	return root

cdef data_struct* divCatDataset(long** idx_data,long attribute,long leftPoints,long dataLen,long n_features,map[long,long] classified) nogil:
	#cdef long e=index_dict[attribute]
	##print "divCatDataset"
	#cdef long e=attribute+3
	cdef long e=attribute+2
	cdef data_struct *split
	cdef long i,j,new_i,new_j
	##print "hee1"
	##print type(n_features)
	##print type(leftPoints)
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
			'''




			if data[idx_data[i][j],e]==1:
				split.left[i][new_j]=idx_data[i][j]
				new_j+=1
			else:
				split.right[i][new_i]=idx_data[i][j]
				new_i+=1
			'''
		##print "hee2"
		if new_j+new_i!=dataLen:
			pass
			#print "problem-hee"+str(new_j+new_i)+"<>"+str(dataLen)
			#with gil:
			#	#print "problem here"+str(new_j+new_i)+"<>"+str(dataLen)
	##print "hee3"
	return split


#cdef 
#data_struct* 
cdef data_struct* divContDataset(long** idx_data,long attribute,double attributeValue,long leftPoints,long dataLen,long n_features,map[long,long] classified) nogil:
	##print "divContDataset"
	#cdef long e=attribute+3
	cdef long e=attribute+2
	cdef data_struct *split
	cdef long i,j,new_i,new_j
	##print "oho1"
	##print n_features,attributeValue,leftPoints,dataLen,attribute
	##print type(n_features)
	##print type(leftPoints)
	
	split=<data_struct *>malloc(sizeof(data_struct))
	split.left=allocate(n_features,leftPoints)
	split.right=allocate(n_features,dataLen-leftPoints)
	##print "oho2"
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
			'''	
			#print (attributeValue,data[idx_data[i][j],e])
			if data[idx_data[i][j],e]<attributeValue:
				split.left[i][new_j]=idx_data[i][j]
				new_j+=1
			else:
				split.right[i][new_i]=idx_data[i][j]
				new_i+=1
			'''
		##print "oho3"+str(j)
		if new_j+new_i!=dataLen:
			pass
			#print "problem-oho"+str(new_j+new_i)+"<>"+str(dataLen)
	##print "oho4"

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


cdef void classify_data(double[:,::1] testData,long isProb,Node *root,long testLen):
	#testData is a list of list
	#return a list of class in order
	#if isProb is True return a list of tuple containing (proba,1-proba)
	#cdef double *testProb=<double *>malloc(sizeof(double)*testLen)
	cdef long i
	cdef double probb
	outfile = open('./new_offerid_opti_home_made_decision_tree', "wb")
	outfile.write('id,repeatProbability\n')
	for i in range(testLen):
		probb=classify_prob(testData[i,:],root,isProb)
		#testProb[i]=classify_prob(testData[i,:],root,isProb)
		outfile.write(str(long(testData[i,0]))+','+str(probb)+'\n')
	outfile.close()
	

def readTest(f):
	#cdef long dataLen
	#cdef double[:,::1] lol
	data = np.genfromtxt(f, dtype=float, delimiter=' ', skip_header=1)
	data=np.delete(data,0,1)#delete label
	data=np.delete(data,0,1)#delete repeat trips
	dataLen=data.shape[0]
	#data=data.astype(np.float32)
	#for e,row in enumerate(data):
	#	testSet.append(list(row))
	return data,dataLen




cdef void printInorder(Node *root):
	if root.isLeaf!=1:
		printInorder(root.left)
	if root.isLeaf==1:	
		print(root.proba)
	if root.isLeaf!=1:
		printInorder(root.right)

# A function to do postorder tree traversal
cdef void printPreorder(Node *root):
		if root.isLeaf==1:
			print (root.proba)
		if root.isLeaf!=1:
			printPreorder(root.left)
			printPreorder(root.right)

##########################################################
# Main of the program
##########################################################
start_time = time.time()
train='./train/train_hot_encoded.csv'
test='./test/test_hot_encoded.csv'
cdef double[::1,:] data
cdef long[::1,:] data_idx_sorted
cdef long[::1] feature_isCat
cdef long num_features,dataLen

#cdef double[:,::1] testSet
#cdef long testLen

cdef Node *tree_root
data,data_idx_sorted,feature_isCat,num_features,dataLen=readData(train)
##print data_idx_sorted
##print data_idx_sorted.ndim
cdef long *feature_i=<long *>malloc(sizeof(long)*num_features)

for k in range(num_features):
	feature_i[k]=k
testSet,testLen=readTest(test)

tree_root=buildDecisionTree(data_idx_sorted,dataLen,15,feature_i,num_features)
#testSet,testLen
#np.savetxt('test.csv',testSet,delimiter=',',newline='\n')
classify_data(testSet,1,tree_root,testLen)
print("--- %s seconds ---" % (time.time() - start_time))
#print ("Pre")
#printPreorder(tree_root)
#print("In..........")
#printInorder(tree_root)














































'''
testSet=readTest(test)
prob_list=classify_data(testSet,True,tree_root)
outfile = open('./new_mem_opti_home_made_random_forest', "wb")
outfile.write('id,repeatProbability\n')
for tup in prob_list:
	outfile.write(str(long(tup[0]))+','+str(tup[1][1])+'\n')
outfile.close()
del tree_root
gc.collect()

'''
#import opti_dec_tree4

'''
d={'chain': 58,
 'has_bought_brand': 42,
 'has_bought_brand_180': 54,
 'has_bought_brand_30': 45,
 'has_bought_brand_60': 48,
 'has_bought_brand_90': 51,
 'has_bought_brand_a': 44,
 'has_bought_brand_a_180': 56,
 'has_bought_brand_a_30': 47,
 'has_bought_brand_a_60': 50,
 'has_bought_brand_a_90': 53,
 'has_bought_brand_category': 7,
 'has_bought_brand_company': 8,
 'has_bought_brand_company_category': 6,
 'has_bought_brand_q': 43,
 'has_bought_brand_q_180': 55,
 'has_bought_brand_q_30': 46,
 'has_bought_brand_q_60': 49,
 'has_bought_brand_q_90': 52,
 'has_bought_category': 27,
 'has_bought_category_180': 39,
 'has_bought_category_30': 30,
 'has_bought_category_60': 33,
 'has_bought_category_90': 36,
 'has_bought_category_a': 29,
 'has_bought_category_a_180': 41,
 'has_bought_category_a_30': 32,
 'has_bought_category_a_60': 35,
 'has_bought_category_a_90': 38,
 'has_bought_category_q': 28,
 'has_bought_category_q_180': 40,
 'has_bought_category_q_30': 31,
 'has_bought_category_q_60': 34,
 'has_bought_category_q_90': 37,
 'has_bought_company': 12,
 'has_bought_company_180': 24,
 'has_bought_company_30': 15,
 'has_bought_company_60': 18,
 'has_bought_company_90': 21,
 'has_bought_company_a': 14,
 'has_bought_company_a_180': 26,
 'has_bought_company_a_30': 17,
 'has_bought_company_a_60': 20,
 'has_bought_company_a_90': 23,
 'has_bought_company_q': 13,
 'has_bought_company_q_180': 25,
 'has_bought_company_q_30': 16,
 'has_bought_company_q_60': 19,
 'has_bought_company_q_90': 22,
 'id': 2,
 'label': 0,
 'market': 57,
 'never_bought_brand': 5,
 'never_bought_category': 4,
 'never_bought_company': 3,
 'offer_id_1190530': 59,
 'offer_id_1194044': 60,
 'offer_id_1197502': 61,
 'offer_id_1198271': 62,
 'offer_id_1198272': 63,
 'offer_id_1198273': 64,
 'offer_id_1198274': 65,
 'offer_id_1198275': 66,
 'offer_id_1199256': 67,
 'offer_id_1199258': 68,
 'offer_id_1200578': 69,
 'offer_id_1200579': 70,
 'offer_id_1200581': 71,
 'offer_id_1200582': 72,
 'offer_id_1200584': 73,
 'offer_id_1200988': 74,
 'offer_id_1203052': 75,
 'offer_id_1203439': 76,
 'offer_id_1204576': 77,
 'offer_id_1204821': 78,
 'offer_id_1204822': 79,
 'offer_id_1208251': 80,
 'offer_id_1208252': 81,
 'offer_id_1208329': 82,
 'offer_id_1208501': 83,
 'offer_id_1208503': 84,
 'offer_id_1213242': 85,
 'offer_id_1219900': 86,
 'offer_id_1219903': 87,
 'offer_id_1220502': 88,
 'offer_id_1220503': 89,
 'offer_id_1221658': 90,
 'offer_id_1221663': 91,
 'offer_id_1221665': 92,
 'offer_id_1221666': 93,
 'offer_id_1221667': 94,
 'offer_id_1230218': 95,
 'offer_value': 9,
 'repeattrips': 1,
 'total_spend_all': 10,
 'total_spend_ccb': 11}



'''

