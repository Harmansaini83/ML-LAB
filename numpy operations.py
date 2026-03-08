import numpy as np   # imports the NumPy library and gives it the alias np


a = np.array([1,2,3,4])   # creates a 1D numpy array with elements 1,2,3,4
print(a)
print(a + 2)   # adds 2 to every element (vectorized addition)
print(a - 2)   # subtracts 2 from every element
print(a * 2)   # multiplies each element by 2 (element-wise)
print(a / 2)   # divides each element by 2 (element-wise, result is float)
print(a ** 2)  # raises each element to the power of 2 (element-wise exponentiation)
print()


b = np.array([1,0,1,0])   # creates another 1D array for element-wise operations
print(a + b)   # element-wise addition between arrays 'a' and 'b'
print(np.sin(a))   # applies sine function to each element of array 'a' (in radians)
print(np.cos(a))   # applies cosine function to each element of array 'a' (in radians)
print(a ** 2)      # squares each element of array 'a' again (same as above)
