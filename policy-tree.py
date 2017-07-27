#coding=utf-8
from math import log


def calcshannoent(dataset):
    numentries = len(dataset)
    labelcounts = {}
    #为所有有可能分类创建字典
    for featvec in dataset:
        currentlabel = featvec[-1]
        print "currentlabel=",currentlabel
        # print "featvec=", featvec
        if currentlabel not in labelcounts.keys():
            print "currentlabel=",currentlabel,"numentries=",numentries
            labelcounts[currentlabel] = 0
            labelcounts[currentlabel] += 1
    shannonrnt = 0.0
    #以2为底求对数
    for key in labelcounts:
            prob = float(labelcounts[key])/numentries
            shannonrnt -= prob * log(prob,2)
    return shannonrnt


def createdataset():
    dataset =[[1,1,'yes'],[1,1,'aa'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset,labels


def main():
    mydata,mylabels = createdataset()
    entropy_i = calcshannoent(mydata)
    print entropy_i


if __name__ == '__main__':
     main()

