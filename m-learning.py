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
    #lisdir���Բ鿴�ļ������ļ�����
    traningfilelist = listdir('trainingDigits')
    m = len(traningfilelist)
    traingmat=zeros((m,1024))
    for i in range(2):
        filenamestr = traningfilelist[i]
        filestr = filenamestr.split(',')[0]
        classnumstr = int(filestr.split('_')[0])
        print "handwrite===",filenamestr,filestr,classnumstr




def classify0(inX,dataset,labels,K):
    ##step1 �������
    datasetsize = dataset.shape[0]
    #datasetsize=900,��Ϊdataset��������100-1000�е����ݣ�����������900
    #tile(A,[2,3])��ʾ����A������������=2*A����=3*A�������ǲ���
    #diffmat����900�еĶ��У���������ÿ�����ݲ�����dataset����һ���Ķ��У�Ȼ���������������룬2�����3���ж�Ӧ���
    diffmat = tile(inX,(datasetsize,1)) -dataset
    # **2������ʽ����ÿ��Ԫ���Լ������Լ�
    sqdiffmat = diffmat**2
    #print "sqdiffmat=**",diffmat[0,:],sqdiffmat[0,:]
    #axis��0��ʾ������ӣ�axis��1��ʾ�����еķ�����ӣ�ÿ����ӹ���900��
    sqdistances = sqdiffmat.sum(axis=1)
    # **0.5���ǿ���
    distances = sqdistances**0.5
    #argsort�ǰ�������ÿ�е�λ���������������[3,6,1],λ��������[0,1,2]��������ֵС����=[2,0,1]
    sorteddistindicies = distances.argsort()
    classcount = {}

    ##step2 ѡ�������С��K����,�����Ѿ���С��������ֻҪȡǰ��K�����Ǿ����������ֵ
    for i in range(K):
        #��������λ����λ���ҵ�labels
        voteilabel = labels[sorteddistindicies[i]]
        #print "voteilabel=",voteilabel
        #dict.get(xx,0)��ͬ�ڻ�ȡ�ֵ��key:value,key��ֵ�������ֵ�ͷ���ֵ��û�оͷ���0,classcount���ڼ���labelֵ���ִ���
        classcount[voteilabel] = classcount.get(voteilabel,0) + 1
    #���ֵ䰴��valueֵ������Ҳ��������дsorted(classcount.items(), key=lambda d: d[1],reverse=True),���Ҳ��dict���list
    sortedclasscount = sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

#���ļ�����ת�ɾ������
def file2matrix(filename):
    readf = open(filename,'r')
    arraylines = readf.readlines()
    #��ȡ�ļ���������Ȼ�󴴽���Ӧ�е�0��
    numrows = len(arraylines)
    ##create matrix numrows ,3 col
    returnmat = zeros((numrows,3))
    classlaber = []
    index = 0
    for line in arraylines:
        #ȥ��ָ���ַ�����Ĭ���ǿ�''
        line = line.strip()
        listfromline = line.split("\t")
        #ȡÿ�е�ǰ3����Ϊ�����һ�У��ļ������ܹ���4�����ݣ���1��ÿ��������������2������Ƶ�ٷֱȣ���3��ÿ�ܱ���ܹ�����
        returnmat[index,:] = listfromline[0:3]
        #����ÿ��line=40920	8.326976	0.953952	3 �����һ��3��ͨ��listfromline[-1]��ȡ
        classlaber.append(int(listfromline[-1]))
        # if index <= 5:
        #     print listfromline,line,"aaa"
        #     print listfromline[-1]
        #     print "returnmat=", returnmat
        #     print "classlaber=", classlaber
        index += 1
    readf.close()
    return returnmat,classlaber

#�Ѿ���תȫ��ת��0-1֮�����ֵ��ȡÿ�������Сֵ��Ȼ��Ը�ֵ-min / max-min���������԰�����ֵ��ת0-1����
def autoNorm(dataset):
    #������1*3������ֵ��3���е���Сֵ�����ֵ
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    #shape��������ȡ������У�������shape[0]=�У�shape[1]=��
    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]
    #tile����1000��1�е�ֵΪminvals�Ķ���
    normdataset = dataset - tile(minvals,(m,1))
    #print "normdataset111=", normdataset,"tile=",tile(minvals,(m,1)),"minvals=",minvals
    normdataset = normdataset/tile(ranges,(m,1))
    #print "normdataset222=", normdataset
    return normdataset,ranges,minvals

def classifyperson():
     resultlist = ["��ϲ��","һ��ϲ��","��ϲ��"]
     percenttats = float(raw_input("percentage of time spent playing video games?"))
     ffmiles = float(raw_input("freguent filer miles earned per year?"))
     icecream = float(raw_input("liters of ice cream consumed per years?"))
     returnmat_r, classlaber_r = file2matrix("D:\Users\yuyan\Desktop\python\m-learning\datingTestSet2.txt")
     normdataset_r, ranges_r, minvals_r = autoNorm(returnmat_r)
     inarr = array([ffmiles,percenttats,icecream])
     classifierresult = classify0((inarr-minvals_r)/ranges_r,normdataset_r,classlaber_r,3)
     print "�㽫������˵ĸо��ǣ�",resultlist[classifierresult -1]

def main():
    returnmat_r,classlaber_r = file2matrix(r"D:\Users\yuyan\Desktop\python\m-learning\datingTestSet2.txt")
    normdataset_r,ranges_r,minvals_r = autoNorm(returnmat_r)
    #�����������վ���Դ���
    horatio = 0.10
    m = normdataset_r.shape[0]
    #�ҳ�������������10%������,1000��*0.10=100��m=1000,numtesetvecs=100
    numtestvecs = int(m*horatio)
    errorcount = 0.0

    for i in range(numtestvecs):
        #ȡ100-1000����numtestvecs��m�У�normdataset_r[numtestvecs:m,:]
        #���ú���classify0��������4������1����ȡÿ��ֱ��100�У���2����100-1000�У���3����100��1000���������࣬��4������3,����ѡ���ھ���Ŀ
        classifieresult = classify0(normdataset_r[i,:],normdataset_r[numtestvecs:m,:],classlaber_r[numtestvecs:m],3)
        print "�Ʋ��� %s,��ʵ���:%d"%(classifieresult,classlaber_r[i])
        if (classifieresult != classlaber_r[i]):
              errorcount += 1.0
    print "�Ʋ����ʵ������:%f"%(errorcount/float(numtestvecs))

    #���������ϢԤ��ϲ�����߲�ϲ��
    #classifyperson()

    #��ӡɢ��ͼ����ͨ�����¾���
    # fig = plt.figure()
    # print fig
    # ax = fig.add_subplot(111)
    # print ax
    # ax.scatter(returnmat_r[:,1],returnmat_r[:,2],15.0*array(classlaber_r),15.0*array(classlaber_r))
    # plt.show()
    #�˴���create_data�ĺ������ý������
    #group_get,lables_get = create_data()
    #print group_get,lables_get
    # aa = create_data()
    # print aa,type(aa)
    # print aa[0],aa[1]

    #ͼƬ
    return2img2da = img2vector(r"D:\Users\yuyan\Desktop\python\m-learning\testDigits\0_9.txt")
    print "return2img2da",return2img2da[0,0:63]
    handwritingclasstest()



if __name__ == '__main__':
     main()

