## **NumPy** 

## **Numerical Python — Complete Notes** 

Theory + Examples + Code | Sahil Pandey | DTU 

## **SECTION 1: Introduction to NumPy** 

## **What is NumPy?** 

NumPy (Numerical Python) is the foundational library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 

## **🔑 Why NumPy over Python Lists?** 

Python lists are slow for numerical operations because they store pointers to objects. NumPy arrays store data in contiguous memory blocks (like C arrays), making operations 10x–100x faster. NumPy also supports vectorized operations — meaning you can operate on entire arrays without Python loops. 

## **Key Features** 

- ndarray — N-dimensional array object 

- Broadcasting — operate on arrays of different shapes 

- Vectorized operations — no explicit loops needed 

- Integration with C/C++/Fortran code 

- Linear algebra, random numbers, Fourier transform 

## **Installation & Import** 

```
# Install
```

```
pip install numpy
# Import convention
import numpy as np
# Check version
print(np.__version__)  # e.g., 1.26.0
```

## **SECTION 2: Creating NumPy Arrays** 

## **NumPy Arrays (ndarray)** 

The ndarray (N-dimensional array) is the core data structure in NumPy. Unlike Python lists, all elements in a NumPy array must be of the same data type (homogeneous). 

## **1. From Python Lists** 

```
import numpy as np
# 1D array from list
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)         # [1 2 3 4 5]
print(type(arr1))   # <class 'numpy.ndarray'>
# 2D array (matrix) from nested list
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]
# 3D array
arr3 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(arr3.shape)   # (2, 2, 2)
```

## **2. Built-in Array Creation Functions** 

## **np.zeros() — Array of all zeros** 

```
np.zeros(5)             # [0. 0. 0. 0. 0.]
```

```
np.zeros((3, 4))        # 3x4 matrix of zeros
```

```
np.zeros((2,3), dtype=int)  # integer zeros
```

## **np.ones() — Array of all ones** 

```
np.ones(4)              # [1. 1. 1. 1.]
np.ones((2, 3))         # 2x3 matrix of ones
```

## **np.full() — Array filled with a constant** 

```
np.full(5, 7)           # [7 7 7 7 7]
```

```
np.full((2,3), 9)       # 2x3 matrix filled with 9
```

## **np.arange() — Like Python range but returns array** 

```
np.arange(10)           # [0 1 2 3 4 5 6 7 8 9]
np.arange(2, 10)        # [2 3 4 5 6 7 8 9]
np.arange(0, 10, 2)     # [0 2 4 6 8]  (step=2)
np.arange(0, 1, 0.1)    # [0.  0.1 0.2 ... 0.9]
```

## **np.linspace() — Evenly spaced over interval** 

```
# 5 equally spaced values between 0 and 1
```

```
np.linspace(0, 1, 5)    # [0.   0.25 0.5  0.75 1.  ]
# 10 values between 0 and 100
np.linspace(0, 100, 10)
```

## **🔑 arange vs linspace** 

Use arange when you want a specific step size. Use linspace when you want a specific number of points. linspace always includes the endpoint by default. 

## **np.eye() — Identity matrix** 

```
np.eye(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

## **np.random — Random arrays** 

```
np.random.seed(42)              # for reproducibility
```

```
np.random.rand(3)               # [0.37 0.95 0.73] — uniform [0,1)
np.random.rand(2, 3)            # 2x3 uniform random
np.random.randn(3)              # standard normal (mean=0, std=1)
np.random.randn(2, 3)           # 2x3 standard normal
np.random.randint(1, 10, 5)     # 5 random ints between 1 and 9
np.random.randint(0, 100, (3,3)) # 3x3 random int matrix
np.random.choice([10,20,30], 5) # pick 5 from given array
```

## **SECTION 3: Array Attributes** 

## **Important Array Attributes** 

Every NumPy array has built-in attributes that tell you about its structure and data type. 

```
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
arr.ndim     # 2  — number of dimensions
arr.shape    # (2, 3)  — (rows, cols)
arr.size     # 6  — total number of elements
arr.dtype    # dtype('int64')  — data type
arr.itemsize # 8  — bytes per element
arr.nbytes   # 48 — total bytes (size * itemsize)
```

## **Data Types (dtype)** 

NumPy supports many data types. Specifying the right dtype saves memory and increases performance. 

```
np.array([1, 2, 3], dtype=np.int8)     # 8-bit integer (-128 to 127)
np.array([1, 2, 3], dtype=np.int32)    # 32-bit integer
np.array([1, 2, 3], dtype=np.int64)    # 64-bit integer (default)
np.array([1.0, 2.0], dtype=np.float32) # 32-bit float
np.array([1.0, 2.0], dtype=np.float64) # 64-bit float (default)
np.array([True, False], dtype=bool)    # boolean
np.array(['a','b'], dtype=np.str_)     # string
# Type conversion
arr = np.array([1.7, 2.9, 3.1])
```

```
arr.astype(int)   # [1 2 3] — truncates, not rounds
```

## **SECTION 4: Indexing & Slicing** 

## **Indexing & Slicing** 

NumPy arrays support powerful indexing — basic (like lists), fancy (with arrays), and boolean. 

## **1D Array Indexing** 

```
arr = np.array([10, 20, 30, 40, 50])
arr[0]      # 10 — first element
arr[-1]     # 50 — last element
arr[1:4]    # [20 30 40] — slice
arr[::2]    # [10 30 50] — every 2nd element
arr[::-1]   # [50 40 30 20 10] — reverse
```

## **2D Array Indexing** 

```
mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
mat[0, 0]    # 1 — row 0, col 0
mat[1, 2]    # 6 — row 1, col 2
mat[-1, -1]  # 9 — last row, last col
# Slicing rows and columns
mat[0, :]    # [1 2 3] — entire first row
mat[:, 1]    # [2 5 8] — entire second column
mat[0:2, 1:3]# [[2 3],[5 6]] — submatrix
# Select specific rows
mat[[0, 2]]  # rows 0 and 2 => [[1,2,3],[7,8,9]]
```

## **Boolean (Conditional) Indexing** 

One of NumPy's most powerful features — filter elements using a condition. 

**==> picture [468 x 237] intentionally omitted <==**

**----- Start of picture text -----**<br>
arr = np.array([10, 25, 30, 15, 40])<br># Boolean mask<br>mask = arr > 20<br>print(mask)       # [False  True  True False  True]<br># Apply mask — returns filtered array<br>arr[mask]         # [25 30 40]<br># In one line<br>arr[arr > 20]     # [25 30 40]<br>arr[arr % 2 == 0] # [10 30 40] — even numbers<br># Modify elements that match condition<br>arr[arr < 20] = 0<br>print(arr)   # [ 0 25 30  0 40]<br>**----- End of picture text -----**<br>


## **Fancy Indexing** 

**==> picture [468 x 133] intentionally omitted <==**

**----- Start of picture text -----**<br>
arr = np.array([100, 200, 300, 400, 500])<br># Index with array of indices<br>idx = [0, 2, 4]<br>arr[idx]     # [100 300 500]<br># 2D fancy indexing<br>mat = np.array([[1,2,3],[4,5,6],[7,8,9]])<br>mat[[0,2], [1,2]]   # [mat[0,1], mat[2,2]] => [2, 9]<br>**----- End of picture text -----**<br>


## **SECTION 5: Reshaping & Manipulation** 

## **Reshaping Arrays** 

## **reshape() — Change shape without changing data** 

```
arr = np.arange(12)  # [0 1 2 3 4 5 6 7 8 9 10 11]
arr.reshape(3, 4)    # 3 rows, 4 cols
```

```
arr.reshape(2, 6)    # 2 rows, 6 cols
arr.reshape(2, 2, 3) # 3D: 2x2x3
# Use -1 to auto-calculate one dimension
arr.reshape(3, -1)   # 3 rows, NumPy figures out cols = 4
arr.reshape(-1, 4)   # NumPy figures out rows = 3
```

## **flatten() and ravel() — Convert to 1D** 

```
mat = np.array([[1,2,3],[4,5,6]])
mat.flatten()  # [1 2 3 4 5 6] — returns a COPY
mat.ravel()    # [1 2 3 4 5 6] — returns a VIEW (faster)
```

## **Transpose** 

```
mat = np.array([[1,2,3],[4,5,6]])  # shape (2,3)
mat.T          # Transpose — shape becomes (3,2)
# [[1 4]
#  [2 5]
#  [3 6]]
np.transpose(mat)  # same result
```

## **Stacking Arrays** 

```
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack([a, b])   # vertical stack => [[1,2,3],[4,5,6]]
np.hstack([a, b])   # horizontal stack => [1,2,3,4,5,6]
np.stack([a, b])    # new axis => [[1,2,3],[4,5,6]]
np.concatenate([a, b])       # [1 2 3 4 5 6]
np.concatenate([a.reshape(1,-1), b.reshape(1,-1)], axis=0)  # 2D stack
```

## **Splitting Arrays** 

```
arr = np.arange(12)
```

```
np.split(arr, 3)     # split into 3 equal parts
```

```
np.hsplit(arr, 4)    # split horizontally
mat = np.arange(16).reshape(4,4)
np.vsplit(mat, 2)    # split vertically into 2
```

## **Adding/Removing Dimensions** 

```
arr = np.array([1, 2, 3])   # shape (3,)
arr[np.newaxis, :]  # shape (1, 3) — row vector
arr[:, np.newaxis]  # shape (3, 1) — column vector
np.expand_dims(arr, axis=0)  # shape (1, 3)
np.expand_dims(arr, axis=1)  # shape (3, 1)
a = np.array([[[1,2,3]]])  # shape (1,1,3)
np.squeeze(a)   # shape (3,) — removes size-1 dimensions
```

## **SECTION 6: Mathematical Operations** 

## **Arithmetic & Math Operations** 

NumPy supports element-wise operations on entire arrays without loops — this is called vectorization. 

## **Element-wise Arithmetic** 

```
a = np.array([10, 20, 30])
b = np.array([2, 4, 5])
a + b     # [12 24 35] — addition
a - b     # [ 8 16 25] — subtraction
a * b     # [20 80 150] — multiplication
a / b     # [5.0 5.0 6.0] — division
a // b    # [5 5 6] — floor division
a % b     # [0 0 0] — modulo
a ** 2    # [100 400 900] — power
# Scalar operations — broadcasts to every element
a + 5     # [15 25 35]
```

```
a * 3     # [30 60 90]
a / 10    # [1. 2. 3.]
```

## **Universal Functions (ufuncs)** 

Universal functions (ufuncs) are NumPy functions that operate element-wise on arrays. 

```
arr = np.array([1, 4, 9, 16, 25])
np.sqrt(arr)    # [1. 2. 3. 4. 5.]
np.square(arr)  # [  1  16  81 256 625]
np.abs(np.array([-1,-2,3]))  # [1 2 3]
np.exp(np.array([0,1,2]))    # [1.    2.718 7.389]
np.log(arr)     # natural log
np.log2(arr)    # log base 2
np.log10(arr)   # log base 10
# Trig functions
angles = np.array([0, np.pi/2, np.pi])
np.sin(angles)  # [0. 1. 0.]
np.cos(angles)  # [1. 0. -1.]
np.tan(angles)  # [0. huge 0.]
# Rounding
arr = np.array([1.4, 1.5, 2.6, 3.7])
np.round(arr)   # [1. 2. 3. 4.]
np.floor(arr)   # [1. 1. 2. 3.]
np.ceil(arr)    # [2. 2. 3. 4.]
```

## **Aggregate / Reduction Functions** 

```
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
np.sum(arr)          # 21 — sum of all elements
np.sum(arr, axis=0)  # [5 7 9] — sum per column
np.sum(arr, axis=1)  # [6 15] — sum per row
np.mean(arr)         # 3.5
np.mean(arr, axis=0) # [2.5 3.5 4.5]
np.min(arr)          # 1
np.max(arr)          # 6
```

**==> picture [468 x 162] intentionally omitted <==**

**----- Start of picture text -----**<br>
np.min(arr, axis=1)  # [1 4]<br>np.max(arr, axis=0)  # [4 5 6]<br>np.argmin(arr)       # 0 — index of minimum<br>np.argmax(arr)       # 5 — index of maximum<br>np.std(arr)          # standard deviation<br>np.var(arr)          # variance<br>np.median(arr)       # 3.5<br>np.cumsum(arr)       # cumulative sum<br>np.prod(arr)         # product of all elements = 720<br>**----- End of picture text -----**<br>


## **SECTION 7: Broadcasting** 

## **Broadcasting** 

Broadcasting allows NumPy to perform operations on arrays of different shapes. NumPy automatically expands the smaller array to match the shape of the larger one — without actually copying data. 

## **🔑 Broadcasting Rules** 

1. If arrays don't have the same number of dims, the shape of the smaller array is padded with 1s on the LEFT. 2. Arrays with size 1 along a dimension are stretched to match the other array. 3. If shapes are incompatible and neither is 1, an error is raised. 

```
# Scalar + Array
arr = np.array([1, 2, 3])
arr + 10    # [11 12 13] — scalar broadcast to array
# 1D + 2D (column broadcast)
a = np.array([[1, 2, 3],   # shape (2,3)
              [4, 5, 6]])
b = np.array([10, 20, 30]) # shape (3,) => treated as (1,3)
a + b
# [[11 22 33]
#  [14 25 36]]
# Column vector + Row vector
col = np.array([[1],[2],[3]])  # shape (3,1)
row = np.array([10, 20, 30])  # shape (1,3) => (3,)
col + row
# [[11 21 31]
```

```
#  [12 22 32]
#  [13 23 33]]
# This would FAIL — shapes (2,3) and (3,2) incompatible
# np.array([[1,2,3],[4,5,6]]) + np.array([[1,2],[3,4],[5,6]])
```

## **SECTION 8: Linear Algebra** 

## **Linear Algebra with NumPy** 

**==> picture [468 x 447] intentionally omitted <==**

**----- Start of picture text -----**<br>
A = np.array([[1, 2],<br>              [3, 4]])<br>B = np.array([[5, 6],<br>              [7, 8]])<br># Matrix multiplication<br>A @ B          # preferred syntax<br>np.dot(A, B)   # same result<br># [[ 19  22]<br>#  [ 43  50]]<br># Element-wise multiplication (NOT matrix multiply)<br>A * B  # [[ 5 12] [21 32]]<br># Determinant<br>np.linalg.det(A)   # -2.0<br># Inverse<br>np.linalg.inv(A)   # [[-2.   1. ][ 1.5 -0.5]]<br># Eigenvalues and eigenvectors<br>vals, vecs = np.linalg.eig(A)<br># Solve linear equation Ax = b<br>b = np.array([5, 11])<br>x = np.linalg.solve(A, b)  # x = [1. 2.]<br># Matrix rank<br>np.linalg.matrix_rank(A)   # 2<br>**----- End of picture text -----**<br>


```
# Trace (sum of diagonal)
np.trace(A)   # 5
```

## **SECTION 9: Comparison & Logical Operations** 

## **Comparison & Logical Operations** 

```
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])
# Element-wise comparison — returns bool array
a == b      # [False False  True False False]
a != b      # [ True  True False  True  True]
a > b       # [False False False  True  True]
a >= b      # [False False  True  True  True]
a < b       # [ True  True False False False]
a <= b      # [ True  True  True False False]
# Logical operations
np.logical_and(a > 2, a < 5)  # [F F T T F]
np.logical_or(a < 2, a > 4)   # [T F F F T]
np.logical_not(a > 3)          # [T T T F F]
# np.where — conditional selection
np.where(a > 3, a, 0)   # [ 0  0  0  4  5]
np.where(a % 2 == 0, 'even', 'odd')  # ['odd','even','odd','even','odd']
# any() and all()
np.any(a > 4)    # True — at least one
np.all(a > 0)    # True — all satisfy
np.all(a > 3)    # False
```

## **SECTION 10: Sorting, Searching & Set Operations** 

## **Sorting** 

```
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
```

```
np.sort(arr)       # [1 1 2 3 4 5 6 9] — returns new array
arr.sort()         # sorts in-place, modifies arr
np.sort(arr)[::-1] # descending order
# argsort — returns indices that would sort the array
arr = np.array([30, 10, 20])
np.argsort(arr)    # [1 2 0] — index 1 is smallest
# Sorting 2D
mat = np.array([[3,1,2],[6,4,5]])
np.sort(mat, axis=1)  # sort each row
np.sort(mat, axis=0)  # sort each column
```

## **Searching** 

```
arr = np.array([10, 20, 30, 40, 50])
np.where(arr == 30)   # (array([2]),) — index of value
np.searchsorted(arr, 25)  # 2 — insertion point
np.nonzero(arr > 25)  # (array([2, 3, 4]),)
```

## **Set Operations** 

```
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])
np.unique(np.array([1,1,2,2,3]))  # [1 2 3]
np.intersect1d(a, b)   # [3 4 5]
np.union1d(a, b)       # [1 2 3 4 5 6 7]
np.setdiff1d(a, b)     # [1 2] — in a but not b
np.in1d(a, b)          # [F F T T T] — element check
```

## **SECTION 11: Copies vs Views** 

## **Copies vs Views** 

**⚠️� Critical Concept!** 

Slicing a NumPy array does NOT create a copy — it creates a VIEW. Modifying a view modifies the original array. Use .copy() to create an independent copy. 

```
original = np.array([1, 2, 3, 4, 5])
# VIEW — shares memory with original
view = original[1:4]
view[0] = 99
print(original)  # [1 99 3 4 5] — original changed!
# COPY — independent array
original = np.array([1, 2, 3, 4, 5])
copy = original[1:4].copy()
copy[0] = 99
print(original)  # [1 2 3 4 5] — original unchanged
# Check if array owns its data
print(view.base is original)   # True — it's a view
print(copy.base is original)   # False — it's a copy
```

## **SECTION 12: Saving & Loading Data** 

## **Saving & Loading Arrays** 

```
arr = np.array([1, 2, 3, 4, 5])
mat = np.array([[1,2],[3,4]])
# Save single array to .npy file
np.save('myarray.npy', arr)
loaded = np.load('myarray.npy')
# Save multiple arrays to .npz file
np.savez('mydata.npz', array1=arr, matrix=mat)
data = np.load('mydata.npz')
data['array1']   # retrieve by key
data['matrix']
# Save to text file (CSV)
np.savetxt('data.csv', mat, delimiter=',', fmt='%d')
loaded = np.loadtxt('data.csv', delimiter=',')
# Load CSV with header skip
data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
```

**QUICK REFERENCE CHEAT SHEET** 

## **NumPy Quick Reference** 

|**Function**|**Description**|
|---|---|
|`np.array()`|Create array from list|
|`np.zeros(n)`|Array of n zeros|
|`np.ones(n)`|Array of n ones|
|`np.arange(start,stop,step`<br>`)`|Range array|
|`np.linspace(start,stop,n)`|n evenly spaced values|
|`np.random.rand()`|Uniform random [0,1)|
|`arr.shape`|Dimensions tuple|
|`arr.dtype`|Data type|
|`arr.reshape(r,c)`|Change shape|
|`arr.flatten()`|Flatten to 1D (copy)|
|`arr.T`|Transpose|
|`np.vstack()`|Stack vertically|
|`np.hstack()`|Stack horizontally|
|`np.sum(arr,axis=)`|Sum (along axis)|
|`np.mean(arr)`|Mean|
|`np.std(arr)`|Standard deviation|
|`np.sort(arr)`|Sort ascending|
|`np.argsort(arr)`|Indices that sort array|
|`np.where(cond,x,y)`|Conditional element pick|
|`np.unique(arr)`|Unique elements|
|`np.dot(A,B) or A@B`|Matrix multiply|
|`np.linalg.inv(A)`|Matrix inverse|



