"""
Numpy Module numpy is far faster than list in python. does not store object type,
object type,reference countand size numpy occupies less space as compared to list numpy 
is more convinient to use.Lists have insertion,deletion,appending,concatenation,etc numpy 
is having lot more than that.

Numpy is faster than lists. Reason: a. List requires object value, object type, reference 
count and size whereas in numpy it directly stores value. b. Lists are scattered and numpy is 
contiguous memory. Numpy occupies less memory as compared to lists. Numpy is more covenient to use. 
Lists have insertion, deletion, appending, concatenation etc. but numpy have lot more than these. 
In terms of applications: Numpy is great in Mathemematics(MATLAB replacement), Plotting (Matplotlib), 
Backend(pandas, connect 4, digital photography), Machine Learning OD=Scalar(int,float,bool)1D=Vector 
2D Matrix/vector of vectprs 3D = tensor/Vector of matrices"""


import numpy as np


a=np.array([(1,2,3),(3,4,4)])
print(a)
print(type(a))#we created array in numpy
print(a.ndim)#dimension number
print(a.shape)
print()


b=np.array((1,2,3))
print(b)
print(b.dtype)#data type of b
print(b.ndim)
print(b.shape)#it helps us to find no of rows and columns
print()


a=np.array([(1,2,3),(3,4,4)],dtype='int16')
print(a.dtype)
print(a.itemsize)
print()


a=np.array([(1,2,3),(3,4,4)],dtype='int32')
print(a.itemsize)
print(a.size)
print(a.shape)
print(a.size *a.itemsize)
print(a.nbytes)
print()

w=np.arange(1,10,2)
print(w)
print()

a=np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
print(a.shape)
print(a[1,4])#accesing specific element
print(a[0,1])#ACCESSING THROUGH NEGATIVE INDEX
#from the same array a i want the elements 2,4,6 to be printed using slicing
print(a[0, [1, 3, 5]])
print()

a=np.zeros((2,2))
print(a)
print()

print(np.zeros((2,3,3)))#dot comes because it automatically takes float value we just have to define
print()

a=np.ones((4,2,2),dtype='int32')
print(a)
print()

print(np.full((2,2),99,dtype='float32'))
print()
