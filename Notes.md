## Python补习

1. 静态方法（**`@staticmethod`**）的使用场景：
   如果在方法中不需要访问任何实例方法和属性，纯粹地通过传入参数并返回数据的功能性方法，那么它就适合用静态方法来定义，它节省了实例化对象的开销成本，往往这种方法放在类外面的模块层作为一个函数存在也是没问题的，而放在类中，仅为这个类服务。
   （相当于定义了一个局部域函数为该类专门服务。）

## Numpy补习

1. **`numpy.array(object, dtype=None)`**
   `object`：创建的数组的对象，可以为单个值，列表，元胞等。
   `dtype`：创建数组中的数据类型。
   返回值：给定对象的数组。

   ```python
   # 普通用法：
   array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
   print("数组array的值为: ")
   print(array)
   print("数组array的默认类型为: ")
   print(array.dtype)
   print("array的类型为: ")
   print(type(array))
   """
   result:
   数组array的值为: 
   [0 1 2 3 4 5 6 7 8 9]
   数组array的默认类型为: 
   int32
   数组array的默认类型为: 
   <class 'numpy.ndarray'>
   创建数组的默认类型为np.int32类型。
   """
   
   # 进阶用法：
   array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
   print("数组array的值为: ")
   print(array)
   print("数组array的默认类型为: ")
   print(array.dtype)
   """
   result:
   数组array的值为: 
   [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
   数组array的默认类型为: 
   float32
   创建数组的默认类型为np.float32类型。
   """
   
   # 更高级的用法：
   array = np.array((1, 2), dtype=[('x', np.int8), ('y', np.int16)])
   print("数组array的值为: ")
   print(array)
   print("数组array的默认类型为: ")
   print(array.dtype)
   print("数组array中对应x标签元素为: ")
   print(array['x'])
   print("数组array中对应y标签元素为: ")
   print(array['y'])
   """
   result:
   数组array的值为: 
   (1, 2)
   数组array的默认类型为: 
   [('x', 'i1'), ('y', '<i2')]
   数组array中对应x标签元素为: 
   1
   数组array中对应y标签元素为: 
   2
   在创建数组的同时，可以设定其中单个元素的数据类型，这里的'i1'指代的便是np.int8类型，'i2'指代的是'np.int16'类型。
   """
   
   # 最高级的用法：
   # Create rain data
   n_drops = 10
   rain_drops = np.zeros(n_drops, dtype=[('position', float, (2,)),
                                         ('size', float),
                                         ('growth', float),
                                         ('color', float, (4,))])
   # Initialize the raindrops in random positions and with
   # random growth rates.
   rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
   rain_drops['growth'] = np.random.uniform(50, 200, n_drops)
   print(rain_drops)
   """
   result:
   [([0.70284885, 0.03590322], 0., 176.4511602 , [0., 0., 0., 0.])
    ([0.60838294, 0.49185854], 0.,  60.51037667, [0., 0., 0., 0.])
    ([0.86525398, 0.65607663], 0., 168.00795695, [0., 0., 0., 0.])
    ([0.25812877, 0.14484747], 0.,  80.17753717, [0., 0., 0., 0.])
    ([0.66021716, 0.90449213], 0., 121.94125106, [0., 0., 0., 0.])
    ([0.88306332, 0.51074725], 0.,  92.4377108 , [0., 0., 0., 0.])
    ([0.68916433, 0.89543162], 0.,  90.77596431, [0., 0., 0., 0.])
    ([0.7105655 , 0.68628326], 0., 144.88783652, [0., 0., 0., 0.])
    ([0.6894679 , 0.90203559], 0., 167.40736266, [0., 0., 0., 0.])
    ([0.92558218, 0.34232054], 0.,  93.48654986, [0., 0., 0., 0.])]
    np.zeros是array的特例。
   """
   
   ```

   

2. **`np.arange()`：**函数**返回**一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是6，步长为1。
   返回类型：<class 'numpy.ndarray'>

   ```python
   #一个参数 默认起点0，步长为1 
   a = np.arange(3)
   # [0 1 2]
   
   #两个参数 默认步长为1 
   a = np.arange(3,9)
   # [3 4 5 6 7 8]
   
   #三个参数 起点为0，终点为3，步长为0.1 
   a = np.arange(0, 3, 0.1)
   # [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.   1.1  1.2  1.3  1.4 1.5  1.6  1.7  1.8  1.9  2.   2.1  2.2  2.3  2.4  2.5  2.6  2.7  2.8  2.9]
   ```

   

3. **`np.random.permutation()`：**随机排列序列。
   返回类型：<class 'numpy.ndarray'>

   ```python
   import numpy as np
   
   # 例1：对0-5之间的序列进行排序
   a = np.random.permutation(5)
   print("a:", a)
   # a: [4 2 0 1 3]
   
   # 例2：对一个list进行排序
   b = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
   print("b:", b)
   # b: [1 3 5 0 2 7 9 6 8 4]
   
   # 例3：对多维数据进行排序
   c1 = np.arange(9).reshape((3,3))
   print("c1:", c1)
   c2 = np.random.permutation(c1)
   print(c2)
   c3 = np.random.permutation(c1)
   print(c3)
   c4 = np.random.permutation(c1)
   print(c4)
   # c1: [[0 1 2]
   #  [3 4 5]
   #  [6 7 8]]
   # c2: [[0 1 2]
   #  [3 4 5]
   #  [6 7 8]]
   # c3: [[3 4 5]
   #  [6 7 8]
   #  [0 1 2]]
   # c4: [[0 1 2]
   #  [3 4 5]
   #  [6 7 8]]
   # 对于一个多维的输入，只是在第一维上进行了随机排序。对这个这个3×3矩阵来说，只是对行进行随机排序。
   ```

4. **numpy.random.shuffle(x)：**修改本身，打乱顺序

   ```python
   import numpy as np
   arr = np.array(range(0, 21, 2))
   np.random.shuffle(arr)
   arr 	
   #打乱顺序后的数组， 如[2, 6, 4, 8, 12, 16, 0, 18, 10, 14, 20]
   
   arr = np.array(range(12)).reshape(3, 4)
   np.random.shuffle(arr)
   arr		
   # 默认对第一维打乱，[[3, 4, 5], [0, 1, 2], [9, 10, 11], [6, 7, 8]]
   ```

5. 对训练集进行打乱

   ```python
   # 数据集：train_X：特征集, train_y：标签
   per = np.random.permutation(train_X.shape[0])		#打乱后的行号
   new_train_X = train_X[per, :]		# 获取打乱后的特征数据
   new_train_y = train_y[per]			# 获取打乱后的标签数据
   ```

6. **`ndarray.shape[ndim]`：**用于指定返回哪一维的长度，返回一个int型结果
   ndim：int型的可选参数，若省略，则返回矩阵在每个维上的长度，返回一个元组。

7. **numpy.empty**(shape, dtype=float, order='C')：根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化。

   | 变量名 |        数据类型        |                             功能                             |
   | :----: | :--------------------: | :----------------------------------------------------------: |
   | shape  | 整数或者整数组成的元组 |             空数组的维度，例如：`(2, 3)`或者`2`              |
   | dtype  |   数值类型，可选参数   | 指定输出数组的数值类型，例如`numpy.int8`。默认为`numpy.float64`。 |
   | order  |  {'C', 'F'}，可选参数  |       是否在内存中以C或fortran(行或列)顺序存储多维数据       |

   返回值：n维数组
   备注：`empty`不像`zeros`一样，并不会将数组的元素值设定为0，因此运行起来可能快一些。在另一方面，它要求用户人为地给数组中的每一个元素赋值，所以应该谨慎使用。
   实例：

   In [1]:

   ```python
   import numpy as np
   np.empty([2, 2])
   ```

   Out[1]:

   ```
   array([[9.90263869e+067, 8.01304531e+262],
          [2.60799828e-310, 0.00000000e+000]])
   ```

   In [2]:

   ```python
   np.empty([2, 2], dtype=int)
   ```

   Out[2]:

   ```
   array([[-1594498784,         506],
          [          0,           0]])
   ```

   

## K-means





## DBCAN