#encoding:utf-8
t=(1,'abc',0.4)
#元组一旦初始化不能改变内部元素
#t[0]=2

l=[1,'abc',0.4]
l[0]=2
l[0]+=1
print l[0]
l[0]-=2
print l[0]