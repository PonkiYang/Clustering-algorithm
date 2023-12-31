2023.12.18	18:00
本项目是模式识别课的第四次实验，基础目标是实现三种典型的聚类方法，完整目标是学习实现所有主流的聚类方法，默写python基础代码，学会使用完整的图表展示程序过程，并编写一个统一的GUI界面，展示所有聚类方法的运作模式。
三种典型聚类方法选择的是K-MEANS方法、DBSCAN方法和层次聚类方法，基于唐宇迪老师的课程：[三小时学懂【三大聚类算法kmeans/DBSCAN/层次】](https://www.bilibili.com/video/BV1ST411w7De/)

下午3点开始做，现在已经6点了，中间吃了顿饭，大概工作时长是两小时，弄明白了k-means和dbscan方法的原理，目前正在看k-means方法的代码。今晚7点就去上课了，是本次实验的最后一课，暂时的任务是，先把三种方法的原理和资料里给的代码顺一遍，弄明白代码基础结构和含义，把代码直接copy到自己的实验项目里面，晚上被老师抽到了好讲给他。之后的背代码和GUI界面等工作等放假了再弄吧，这几天实在没时间。

另外，这估计是我认真准备并学习的第一个项目（之前的实验都是一路水过来的，你问学到了什么，可谓是什么都没有学到），先上传下Github，前几天第三次学的git使用方法又快忘干净了！

18:30
弄了半天，终于给github提交上了，也顺便复习了下前几天学的git用法，这样就不用copy文件在两个电脑之间传来传去，去实验室直接clone就可以了。

18:58
两份代码大概都弄明白了，上传一下，去实验室了。

20:21
因为唐宇迪老师讲DBSCAN的时候写的代码是直接调用的DBSCAN库，并不像k-means一样直接用python写，所以我另外又找了一些DBSCAN的资料，发现基本上全都是调库来的。于是我直接找到库里面看了一眼类的原型，发现里面好多看不懂的代码，也调用了好多没见过的东西，好在内容并不是特别多，去除注释后代码一共100行多一些，之后再研究吧。
DBSCAN.py参考自：[DBSCAN聚类的python实现_python dbscan-CSDN博客](https://blog.csdn.net/qq_18055167/article/details/128493668)
DBSCAN_1.py参考自：[【机器学习】DBSCAN聚类算法（含Python实现）_dbscan python-CSDN博客](https://blog.csdn.net/wzk4869/article/details/129775584)
DBSCAN_2.py参考自：[鸢尾花三种聚类算法（K-means,AGNES,DBScan）的python实现-CSDN博客](https://blog.csdn.net/weixin_42134141/article/details/80413598)
之后闲下来再把这几篇文过一遍。

20:35
老师说今天我们班里挺多人咳嗽的，就不抽人讲了，好耶！（因为突然发现还有个东西没看懂，就那个select_MinPts()函数，根据数据集和选定的k值返回核心对象的最小样本数，k值我也没弄明白是啥）现在就先把实验报告写了吧，尽量在9点前写完，然后直接偷偷溜走回宿舍了。

另外突然发现单片机数码管的实验报告截止今天12点，回宿舍赶紧抄下室友的交上去。
另外记得之后把几篇参考材料copy下来。

21:30
焯，弄了将近1个小时还没弄完，回宿舍弄吧。

22:30
一个更深入的k-means算法解析，抽空看：[聚类（K-means、K-均值）算法的基础、原理、Python实现和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/158776162)
弄完实验报告了，算是勉勉强强混过去了，但详细的内容还得放假后重新认真弄一遍，估计要花上一整天的时间。

整理了一下大概任务：

1. 看完[我居然三小时学懂了【三大聚类算法kmeans/DBSCAN/层次】，不愧是清华计算机博士亲授课程！—人工智能/机器学习/聚类分析_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1ST411w7De/)
2. 看完以下几篇文章，并copy文章内容到项目文件夹中：
   [DBSCAN聚类的python实现_python dbscan-CSDN博客](https://blog.csdn.net/qq_18055167/article/details/128493668)
   [【机器学习】DBSCAN聚类算法（含Python实现）_dbscan python-CSDN博客](https://blog.csdn.net/wzk4869/article/details/129775584)
   [鸢尾花三种聚类算法（K-means,AGNES,DBScan）的python实现-CSDN博客](https://blog.csdn.net/weixin_42134141/article/details/80413598)
   [聚类（K-means、K-均值）算法的基础、原理、Python实现和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/158776162)
3. 记熟两种算法的代码，要求会默写
4. 实现GUI图形界面，类似：[Visualizing K-Means Clustering (naftaliharris.com)](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)