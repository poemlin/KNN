from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

#######
# 数据导入模块
#######

def dataload(filename):
    # fr成了一个打开文件的对象
    fr = open(filename)
    # 这个对象有个readlines函数，一次读取文件所有内容，并返回一个list，每个元素一行
    lines = fr.readlines()
    # 定义最后存放标签的list
    labels = []
    # 返回list的大小，即文件的行数
    linesSize = len(lines)
    # 初始化一个数组，用于存放读入的样本
    returnMat = zeros((linesSize,3))
    # index记录行数
    index = 0
    # 对每一行进行for循环操作
    for line in lines:
        # 去除末尾的换行
        line = line.strip()
        # 把每一行以tap分割
        inline = line.split("\t")
        # 把每一行的数据逐行放入数组，注意切片操作
        returnMat[index,:] = inline[:-1]
        # 最后一个元素放入标签列表，注意int（读入的是string）
        labels.append(int(inline[-1]))
        index+=1

    return returnMat,labels

######
#k近邻主算法
######

def knn(inx,dataset,labels,k):
    # 获得样本数组的行数
    dataSize = dataset.shape[0]
    # 把输入的新样本重复行数次，生成一个和样本数组一样大小的数组
    inxMat = tile(inx, (dataSize,1))
    # 重复数组减去样本数组
    subtrainxMat = inxMat - dataset
    # 平方
    squareinxMat = subtrainxMat**2
    # 按行求和
    sqDistance = squareinxMat.sum(axis=1)
    # argsort排序后返回元素的序号
    sortDistance = sqDistance.argsort()
    # 生成 标签+出现次数 的字典
    classcount = {}
    # 最近的几个样本
    for i in range(k):
        votelabel = labels[sortDistance[0]]
        classcount[votelabel] = classcount.get(votelabel,0) + 1
    # sorted函数 要排序的东西 key排序规则 reverse正反顺序 True大写
    sortedclasscount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedclasscount[0][0]

#####
#创建所有数据的散点图
#####

def pictureshow(dataset):
    # 创建一幅图
    fig = plt.figure()
    # 图1*1的第一块
    ax = fig.add_subplot(111)
    # 创建散点图，X,Y分别是第一列和第二列数据
    ax.scatter(dataset[:,1],dataset[:,2])
    plt.show()

#####
#归一化样本模块
#####

def datanorm(dataset):
    # 按列求每列的最小值，结果放入一个list
    minvec = dataset.min(0)
    # print(minvec)
    # 按列的原因是，每列的最大值和最小值才有意义，列对应一个属性或一个维度
    maxvec = dataset.max(0)
    # 初始化归一化后的数据集
    normdataset = zeros(shape(dataset))
    # 样本的个数（行数）
    m = dataset.shape[0]
    # 最小和最大list各自重复行数遍，形成和样本集一样大小的数组
    minset = tile(minvec,(m,1))
    maxset = tile(maxvec,(m,1))
    # 归一化公式
    normdataset = (dataset - minset)/(maxset - minset)
    return normdataset




#######
#测试正确率模块
#######
def knntest(dataset,labels,odds,k):
    # 首先归一化数据集
    normdataset = datanorm(dataset)
    # print(normdataset)
    # 仍然计算样本个数
    m = normdataset.shape[0]
    # print(m)
    # 用于测试的样本个数
    numTest = int(m*odds)
    # 初始化正确样本个数
    rightCount = 0.0
    for i in range(numTest):
        # 前numTest个样本用于测试，后面的用做训练集
        knnresult = knn(normdataset[1,:],normdataset[numTest:m,:],labels[numTest:m],k)
        # print(knnresult)
        # print(labels[i])
        if(knnresult == labels[i]):
            rightCount+=1.0

    print("测试样本数目：%d ; 正确预测分类个数：%d \n" % (numTest,rightCount))
    print("分类正确率：%f" % (rightCount/float(numTest)))

#######
#数据模块
#######
dataset,datasetLabels=dataload("knndata.txt")
knntest(dataset,datasetLabels,0.05,3)
pictureshow(dataset)