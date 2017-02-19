'''
1. optiimized version of decision tree earlier built
2. uses presort for fast split finding
3. data now stored as list of list where each inner list is data sorted on the basis of continuous variable.
4. also sends cols where spliit possible to each node instead of keeping it global, this makes same split in cousins and brothers possible
5. still using depth first to build the tree
6. still no pruning included
'''
import numpy as np
from sets import Set
import math
import gc

class Node:
	def __init__(self):
		self.height=None
		self.attribute=None
		self.isCategorical=None
		self.attributeValue=None
		self.infoGain=None
		self.left=None
		self.right=None
		self.data=None
		#prob of being in class 0
		self.proba=None
		self.entropy=None	#current entropy
		self.availableAtt=None

def listCol():
	'''
	Return a list of columns/attributes to split
	'''
	col=['id','offer_id','never_bought_company','never_bought_category','never_bought_brand','has_bought_brand_company_category','has_bought_brand_category','has_bought_brand_company','offer_value','total_spend_all', 'total_spend_ccb','has_bought_company','has_bought_company_q','has_bought_company_a','has_bought_company_30','has_bought_company_q_30','has_bought_company_a_30','has_bought_company_60','has_bought_company_q_60','has_bought_company_a_60',
'has_bought_company_90','has_bought_company_q_90','has_bought_company_a_90','has_bought_company_180','has_bought_company_q_180','has_bought_company_a_180','has_bought_category','has_bought_category_q', 'has_bought_category_a','has_bought_category_30','has_bought_category_q_30','has_bought_category_a_30','has_bought_category_60','has_bought_category_q_60','has_bought_category_a_60','has_bought_category_90', 'has_bought_category_q_90','has_bought_category_a_90','has_bought_category_180','has_bought_category_q_180','has_bought_category_a_180','has_bought_brand','has_bought_brand_q','has_bought_brand_a','has_bought_brand_30',
'has_bought_brand_q_30','has_bought_brand_a_30','has_bought_brand_60','has_bought_brand_q_60','has_bought_brand_a_60','has_bought_brand_90','has_bought_brand_q_90','has_bought_brand_a_90','has_bought_brand_180', 'has_bought_brand_q_180','has_bought_brand_a_180','market','chain']
	col.remove('id')
	col.remove('offer_id')
	return col


def dictCol():
	'''
	Return a dict of col for continuous valued data or categorical data
	dict{key:val}==={attribute(index):0/1(1 means categorical)}
	'''
	d={'chain': 0,
 'has_bought_brand': 0,
 'has_bought_brand_180': 0,
 'has_bought_brand_30': 0,
 'has_bought_brand_60': 0,
 'has_bought_brand_90': 0,
 'has_bought_brand_a': 0,
 'has_bought_brand_a_180': 0,
 'has_bought_brand_a_30': 0,
 'has_bought_brand_a_60': 0,
 'has_bought_brand_a_90': 0,
 'has_bought_brand_category': 1,
 'has_bought_brand_company': 1,
 'has_bought_brand_company_category': 1,
 'has_bought_brand_q': 0,
 'has_bought_brand_q_180': 0,
 'has_bought_brand_q_30': 0,
 'has_bought_brand_q_60': 0,
 'has_bought_brand_q_90': 0,
 'has_bought_category': 0,
 'has_bought_category_180': 0,
 'has_bought_category_30': 0,
 'has_bought_category_60': 0,
 'has_bought_category_90': 0,
 'has_bought_category_a': 0,
 'has_bought_category_a_180': 0,
 'has_bought_category_a_30': 0,
 'has_bought_category_a_60': 0,
 'has_bought_category_a_90': 0,
 'has_bought_category_q': 0,
 'has_bought_category_q_180': 0,
 'has_bought_category_q_30': 0,
 'has_bought_category_q_60': 0,
 'has_bought_category_q_90': 0,
 'has_bought_company': 0,
 'has_bought_company_180': 0,
 'has_bought_company_30': 0,
 'has_bought_company_60': 0,
 'has_bought_company_90': 0,
 'has_bought_company_a': 0,
 'has_bought_company_a_180': 0,
 'has_bought_company_a_30': 0,
 'has_bought_company_a_60': 0,
 'has_bought_company_a_90': 0,
 'has_bought_company_q': 0,
 'has_bought_company_q_180': 0,
 'has_bought_company_q_30': 0,
 'has_bought_company_q_60': 0,
 'has_bought_company_q_90': 0,
# 'id': 0,
 'market': 0,
 'never_bought_brand': 1,
 'never_bought_category': 1,
 'never_bought_company': 1,
# 'offer_id': 0,
 'offer_value': 0,
 'total_spend_all': 0,
 'total_spend_ccb': 0}
	return d

def indict():
	d={'chain': 57,
 'has_bought_brand': 41,
 'has_bought_brand_180': 53,
 'has_bought_brand_30': 44,
 'has_bought_brand_60': 47,
 'has_bought_brand_90': 50,
 'has_bought_brand_a': 43,
 'has_bought_brand_a_180': 55,
 'has_bought_brand_a_30': 46,
 'has_bought_brand_a_60': 49,
 'has_bought_brand_a_90': 52,
 'has_bought_brand_category': 6,
 'has_bought_brand_company': 7,
 'has_bought_brand_company_category': 5,
 'has_bought_brand_q': 42,
 'has_bought_brand_q_180': 54,
 'has_bought_brand_q_30': 45,
 'has_bought_brand_q_60': 48,
 'has_bought_brand_q_90': 51,
 'has_bought_category': 26,
 'has_bought_category_180': 38,
 'has_bought_category_30': 29,
 'has_bought_category_60': 32,
 'has_bought_category_90': 35,
 'has_bought_category_a': 28,
 'has_bought_category_a_180': 40,
 'has_bought_category_a_30': 31,
 'has_bought_category_a_60': 34,
 'has_bought_category_a_90': 37,
 'has_bought_category_q': 27,
 'has_bought_category_q_180': 39,
 'has_bought_category_q_30': 30,
 'has_bought_category_q_60': 33,
 'has_bought_category_q_90': 36,
 'has_bought_company': 11,
 'has_bought_company_180': 23,
 'has_bought_company_30': 14,
 'has_bought_company_60': 17,
 'has_bought_company_90': 20,
 'has_bought_company_a': 13,
 'has_bought_company_a_180': 25,
 'has_bought_company_a_30': 16,
 'has_bought_company_a_60': 19,
 'has_bought_company_a_90': 22,
 'has_bought_company_q': 12,
 'has_bought_company_q_180': 24,
 'has_bought_company_q_30': 15,
 'has_bought_company_q_60': 18,
 'has_bought_company_q_90': 21,
 'id': 0,
 'market': 56,
 'never_bought_brand': 4,
 'never_bought_category': 3,
 'never_bought_company': 2,
 'offer_id': 1,
 'offer_value': 8,
 'total_spend_all': 9,
 'total_spend_ccb': 10}


	return d


def readData(f):
	'''
	Reads data from file into numpy
	Returns a list of list of tuples
	list_order--->get the 
	tuple--->(point,label)
	point---->list of features
	'''
	dataSet=[]
	preSorted=[]
	list_order=[]
	data = np.genfromtxt(f, dtype=float, delimiter=' ', skip_header=1)
	labels=data[:,0]
	#print (labels)
	data=np.delete(data,0,1)
	data=np.delete(data,0,1)
	
	#dropping id and offer_id for meantime
	#data=np.delete(data,0,1)
	#data=np.delete(data,0,1)

	for e,row in enumerate(data):
		#pass
		#print e
		dataSet.append((list(row),int(labels[e])))
	#this dataset contains list of tuples
	#print dataSet
	#return dataSet
	listMmap={}
	##change1
	cdef int iind=0
	for i in range(len(col)):
		if col_dict[col[i]]==0.0:
			y=sorted(dataSet,key=lambda tup:tup[0][i+2])
			preSorted.append(y)
			list_order.append(col[i])
			listMmap[col[i]]=iind
			iind+=1
	return (list_order,preSorted,listMmap)


def entropy(prob_tup):
	#print prob_tup[0],prob_tup[1]
	if prob_tup[0]==0 or prob_tup[1]==0:
		return 0.0
	return (-prob_tup[0]*math.log(prob_tup[0],2)-prob_tup[1]*math.log(prob_tup[1],2))


def calProb(data):
	cdef float countp=0.0
	cdef float countn=0.0
	#print len(data)
	if len(data)==0:
		return (0.0,0.0)

	for i in data:
		if i[1]==1:
			countp+=1.0
		else:
			countn+=1.0
	return (countp/len(data),countn/len(data))


def calCatInfo(data,att,e,currEntropy,dataLen):
	##here the data is sorted list of tups 
	trueList=[]
	falseList=[]
	print "hello8"
	for i in data:
		if i[0][e]==1:
			trueList.append(i)
		else:
			falseList.append(i)
	#print len(trueList),len(falseList)
	trueEntropy=entropy(calProb(trueList))
	falseEntropy=entropy(calProb(falseList))
	trueProb=float(len(trueList))/dataLen
	falseProb=float(len(falseList))/dataLen
	expectedEntropy=trueProb*trueEntropy+falseProb*falseEntropy
	infoGain=currEntropy-expectedEntropy
	#return [infoGain,trueList,falseList]
	return [infoGain,None]


def calContInfo(data,att,e,currEntropy,dataLen):
	##here the data is sorted list of tups corresponding to att
	#currEntropy=entropy(calProb(data))
	cdef float infoGain=-1
	cdef float value=data[0][0][e]
	cdef float leftNeg=0.0
	cdef float leftPos=0.0
	cdef float rightNeg=0.0
	cdef float now_class=data[0][1]
	cdef float skipPos=0.0
	cdef float skipNeg=0.0
	cdef float trueProb, falseProb, trueEntropy, falseEntropy, iGain, expectedEntropy

	for i in data:
		if i[1]==0.0:
			rightNeg+=1.0
	rightPos=dataLen-rightNeg
	for i in range(1,dataLen):
		if data[i][1]==now_class:
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
			trueProb=(leftNeg+leftPos)/float(dataLen)
			falseProb=(rightNeg+rightPos)/float(dataLen)
			trueEntropy=entropy((leftPos/(leftPos+leftNeg),leftNeg/(leftPos+leftNeg)))
			falseEntropy=entropy((rightPos/(rightPos+rightNeg),rightNeg/(rightPos+rightNeg)))
			expectedEntropy=trueProb*trueEntropy+falseProb*falseEntropy
			iGain=currEntropy-expectedEntropy
			if infoGain<iGain:
				infoGain=iGain
				value=data[i][0][e]
	return [infoGain,value]


def bestFeature(data,probTup,avail_attri):
	##here the data is list of list 
	cdef float max_info_gain=-1
	value=None
	attribute=None
	cdef float currEntropy=entropy(probTup)
	cdef int e
	##can be taken in func defn
	cdef float dataLen=len(data[0])
	for att in avail_attri:
		#print att
		e=index_dict[att]
		if col_dict[att]==1:
			a=calCatInfo(data[0],att,e,currEntropy,dataLen)
			flag=True
		else:
			list_indd=list_order_map[att]
			a=calContInfo(data[list_indd],att,e,currEntropy,dataLen)
			flag=False
		if a[0]>=max_info_gain:
			max_info_gain=a[0]
			if flag:
				isCategorical=1
				value=None
				attribute=att
			else:
				isCategorical=0
				value=a[1]
				attribute=att
	return [max_info_gain,isCategorical,value,attribute]


def divCatDataset(data,attribute):
	cdef int e=index_dict[attribute]
	trueList=[]
	falseList=[]
	for i in data:
		leftList=[]
		rightList=[]
		for j in i:
			if j[0][e]==1:
				leftList.append(j)
			else:
				rightList.append(j)
		trueList.append(leftList)
		falseList.append(rightList)
	return (trueList,falseList)


def divContDataset(data,attribute,attributeValue):
	cdef int e=index_dict[attribute]
	trueList=[]
	falseList=[]
	for i in data:
		leftList=[]
		rightList=[]
		for j in i:
			if j[0][e]<attributeValue:
				leftList.append(j)
			else:
				rightList.append(j)
		trueList.append(leftList)
		falseList.append(rightList)
	return (trueList,falseList)



def buildDecisionTree(data,max_height,avail_attri):
	##here the data is list of list
	##delete data to be added
	fflag=False
	root=Node()
	root.data=data
	root.height=1
	curr=root
	#print curr.data
	stack=[]
	stack.append(curr)
	curr.availableAtt=avail_attri
	while(len(stack)!=0):
		if fflag:
			gc.collect()
		print "hello"
		attList1=[]
		attList2=[]
		curr=stack.pop()
		probTup=calProb(curr.data[0])
		#isSameTup=isSameLabels(curr.data)
		if probTup[0]==0.0:
			print "hello1"
			curr.proba=1.0
			curr.isLeaf=1
			fflag=False
		elif probTup[1]==0.0:
			curr.proba=0.0
			curr.isLeaf=1
			fflag=False
		elif len(curr.availableAtt)==0:
			print "hello2"
			curr.proba=probTup[1]
			curr.isLeaf=1
			fflag=False
		elif curr.height==max_height:
			print "hello3"
			curr.proba=probTup[1]
			curr.isLeaf=1
			fflag=False
		else:
			fflag=True
			print "hello4"
			print "hello honey "+str(len(curr.availableAtt))+str(curr.height)
			info=bestFeature(curr.data,probTup,curr.availableAtt)
			curr.infoGain=info[0]
			curr.isCategorical=info[1]
			curr.attributeValue=info[2]
			curr.attribute=info[3]
			if curr.infoGain==0:
				print "hello5"
				curr.proba=probTup[1]
				curr.isLeaf=1
			else:
				print "hello6"
				#for attt in curr.availableAtt:
				#	if attt!=curr.attribute:
				#		attList1.append(attt)
				#		attList2.append(attt)
				leftChild=Node()
				rightChild=Node()
				if curr.isCategorical==1:
					leftList,rightList=divCatDataset(data,curr.attribute)
				else:
					leftList,rightList=divContDataset(data,curr.attribute,curr.attributeValue)
				leftChild.data=leftList
				rightChild.data=rightList
				leftChild.height=curr.height+1
				rightChild.height=curr.height+1
				##remove if needed
				curr.availableAtt.remove(curr.attribute)
				#leftChild.availableAtt=attList1
				#rightChild.availableAtt=attList1
				leftChild.availableAtt=curr.availableAtt
				rightChild.availableAtt=curr.availableAtt
				curr.isLeaf=0
				curr.left=leftChild
				curr.right=rightChild
				stack.append(curr.right)
				stack.append(curr.left)
			del curr.data
	return root

def classify_prob(dataPoint,root,isProb):
	#Here dataPoint is a list of features
	curr=root
	while(True):
		if curr.isLeaf==1:
			if isProb:
				return (curr.proba,1-curr.proba)
			else:
				if curr.proba>=0.5:
					return 0
				else:
					return 1
		else:
			index=index_dict[curr.attribute]
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


def classify_data(testData,isProb,root):
	#testData is a list of list
	#return a list of class in order
	#if isProb is True return a list of tuple containing (proba,1-proba)
	llist=[]
	for i in testData:
		llist.append((i[0],classify_prob(i,root,isProb)))
	return llist


def readTest(f):
	testSet=[]
	data = np.genfromtxt(f, dtype=float, delimiter=' ', skip_header=1)
	data=np.delete(data,0,1)
	data=np.delete(data,0,1)
	for e,row in enumerate(data):
		testSet.append(list(row))
	return testSet



##########################################################
# Main of the program
##########################################################
gc.enable()
train='./train/noreduce_train_base_features.csv'
test='./test/test_base_features.csv'
col=listCol()
index_dict=indict()
col_dict=dictCol()
list_order,trainSet,list_order_map=readData(train)
testSet=readTest(test)
tree_root=buildDecisionTree(trainSet,15,col)
prob_list=classify_data(testSet,True,tree_root)
outfile = open('./new_opti_home_made_random_forest', "wb")
outfile.write('id,repeatProbability\n')
for tup in prob_list:
	outfile.write(str(tup[0])+','+str(tup[1][1])+'\n')
outfile.close()




