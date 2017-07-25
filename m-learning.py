#coding=utf-8
import numpy
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import *


def create_data():
    group = array([[1,0,1,1],[1,0,1,0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group,lables


def img2vector(filename):
    returnvect = zeros((1,1024))
    fr = open(filename,'r')
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j] = int(linestr[j])
    return returnvect


def handwritingclasstest():
    hwlabels = []
    #lisdir可以查看文件里面文件数量
    traningfilelist = listdir('trainingDigits')
    m = len(traningfilelist)
    traingmat=zeros((m,1024))
    for i in range(2):
        filenamestr = traningfilelist[i]
        filestr = filenamestr.split(',')[0]
        classnumstr = int(filestr.split('_')[0])
        print "handwrite===",filenamestr,filestr,classnumstr




def classify0(inX,dataset,labels,K):
    ##step1 计算距离
    datasetsize = dataset.shape[0]
    #datasetsize=900,因为dataset传过来是100-1000行的数据，所以总数是900
    #tile(A,[2,3])表示按照A的样子生成行=2*A，列=3*A，这里是产生
    #diffmat等于900行的队列，传过来的每行数据产生跟dataset行数一样的队列，然后相减，用于算距离，2组队列3个列对应相减
    diffmat = tile(inX,(datasetsize,1)) -dataset
    # **2等行列式里面每个元素自己乘以自己
    sqdiffmat = diffmat**2
    #print "sqdiffmat=**",diffmat[0,:],sqdiffmat[0,:]
    #axis＝0表示按列相加，axis＝1表示按照行的方向相加，每行相加共有900行
    sqdistances = sqdiffmat.sum(axis=1)
    # **0.5即是开方
    distances = sqdistances**0.5
    #argsort是按照数列每行的位置来排序，例如队列[3,6,1],位置索引是[0,1,2]，按照数值小到大=[2,0,1]
    sorteddistindicies = distances.argsort()
    classcount = {}

    ##step2 选择距离最小的K个点,距离已经由小到大排序，只要取前面K个就是距离最近的数值
    for i in range(K):
        #根据索引位置行位置找到labels
        voteilabel = labels[sorteddistindicies[i]]
        #print "voteilabel=",voteilabel
        #dict.get(xx,0)等同于获取字典的key:value,key的值，如果有值就返回值，没有就返回0,classcount用于计算label值出现次数
        classcount[voteilabel] = classcount.get(voteilabel,0) + 1
    #把字典按照value值来排序，也可以这样写sorted(classcount.items(), key=lambda d: d[1],reverse=True),结果也由dict变成list
    sortedclasscount = sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

#把文件内容转成矩阵队列
def file2matrix(filename):
    readf = open(filename,'r')
    arraylines = readf.readlines()
    #获取文件的行数，然后创建对应行的0阵
    numrows = len(arraylines)
    ##create matrix numrows ,3 col
    returnmat = zeros((numrows,3))
    classlaber = []
    index = 0
    for line in arraylines:
        #去掉指定字符串，默认是空''
        line = line.strip()
        listfromline = line.split("\t")
        #取每行的前3个作为矩阵的一行，文件数据总共有4列数据，第1列每年飞行里程数，第2列玩视频百分比，第3列每周冰淇淋公升数
        returnmat[index,:] = listfromline[0:3]
        #数据每行line=40920	8.326976	0.953952	3 ，最后一个3是通过listfromline[-1]获取
        classlaber.append(int(listfromline[-1]))
        # if index <= 5:
        #     print listfromline,line,"aaa"
        #     print listfromline[-1]
        #     print "returnmat=", returnmat
        #     print "classlaber=", classlaber
        index += 1
    readf.close()
    return returnmat,classlaber

#把矩阵转全部转成0-1之间的数值，取每列最大，最小值，然后对个值-min / max-min，这样可以把所有值都转0-1区间
def autoNorm(dataset):
    #返回是1*3的数组值，3个列的最小值，最大值
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    #shape的作用是取矩阵的行，列数，shape[0]=行，shape[1]=列
    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]
    #tile生成1000行1列的值为minvals的队列
    normdataset = dataset - tile(minvals,(m,1))
    #print "normdataset111=", normdataset,"tile=",tile(minvals,(m,1)),"minvals=",minvals
    normdataset = normdataset/tile(ranges,(m,1))
    #print "normdataset222=", normdataset
    return normdataset,ranges,minvals

def classifyperson():
     resultlist = ["不喜欢","一点喜欢","很喜欢"]
     percenttats = float(raw_input("percentage of time spent playing video games?"))
     ffmiles = float(raw_input("freguent filer miles earned per year?"))
     icecream = float(raw_input("liters of ice cream consumed per years?"))
     returnmat_r, classlaber_r = file2matrix("D:\Users\yuyan\Desktop\python\m-learning\datingTestSet2.txt")
     normdataset_r, ranges_r, minvals_r = autoNorm(returnmat_r)
     inarr = array([ffmiles,percenttats,icecream])
     classifierresult = classify0((inarr-minvals_r)/ranges_r,normdataset_r,classlaber_r,3)
     print "你将对这个人的感觉是：",resultlist[classifierresult -1]

def main():
    returnmat_r,classlaber_r = file2matrix(r"D:\Users\yuyan\Desktop\python\m-learning\datingTestSet2.txt")
    normdataset_r,ranges_r,minvals_r = autoNorm(returnmat_r)
    #分类器针对网站测试代码
    horatio = 0.10
    m = normdataset_r.shape[0]
    #找出数据总行数的10%来测试,1000行*0.10=100，m=1000,numtesetvecs=100
    numtestvecs = int(m*horatio)
    errorcount = 0.0

    for i in range(numtestvecs):
        #取100-1000，即numtestvecs到m行，normdataset_r[numtestvecs:m,:]
        #调用函数classify0，参数传4个，第1个读取每行直到100行，第2个是100-1000行，第3个是100至1000行样本分类，第4个等于3,用于选择邻居数目
        classifieresult = classify0(normdataset_r[i,:],normdataset_r[numtestvecs:m,:],classlaber_r[numtestvecs:m],3)
        print "推测结果 %s,现实结果:%d"%(classifieresult,classlaber_r[i])
        if (classifieresult != classlaber_r[i]):
              errorcount += 1.0
    print "推测和现实错误率:%f"%(errorcount/float(numtestvecs))

    #输入对象信息预测喜欢或者不喜欢
    #classifyperson()

    #打印散点图就是通过如下就行
    # fig = plt.figure()
    # print fig
    # ax = fig.add_subplot(111)
    # print ax
    # ax.scatter(returnmat_r[:,1],returnmat_r[:,2],15.0*array(classlaber_r),15.0*array(classlaber_r))
    # plt.show()
    #此处是create_data的函数调用结果测试
    #group_get,lables_get = create_data()
    #print group_get,lables_get
    # aa = create_data()
    # print aa,type(aa)
    # print aa[0],aa[1]

    #图片
    return2img2da = img2vector(r"D:\Users\yuyan\Desktop\python\m-learning\testDigits\0_9.txt")
    print "return2img2da",return2img2da[0,0:63]
    handwritingclasstest()



if __name__ == '__main__':
     main()

