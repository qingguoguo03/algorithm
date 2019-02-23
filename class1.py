
# 主要就是各种排序与例题 Python的实现版本
import random
def return_random_arr():
    arr = [ random.randint(0,1000) for i in range(random.randint(1, 50))]
    print(arr)
    return arr

def check(arr):
    for i in range(len(arr)-1):
        if arr[i]>arr[i+1]:
            raise
    return True

def bubbleSort(arr):
    for i in range(0, len(arr)-1):
        for j in range(0, len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
bubbleSort(arr)

def select_sort(arr):
    for i in range(0, len(arr)-1):
        for j in range(i+1, len(arr)):
            if arr[i]>arr[j]:
                arr[j], arr[i] = arr[i], arr[j]  
select_sort(arr)

def insert_sort(arr):
    for i in range(1, len(arr)):
        for j in range(i, 0, -1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
            else:
                break
insert_sort(arr)
check(arr)
print(arr)
    

def merge_sort(arr):
    def merge(a, b):
        c = []
        i = j = 0 
        while i < len(a) and j < len(b):
            if a[i]>=b[j]:
                c.append(b[j])
                j += 1
            else:
                c.append(a[i])
                i += 1
        c =  c + a[i:] + b[j:] # 这里生产了一个list，视频中在这个之后对arr进行了赋值，所有的操作以index进行处理
        return c
    
    def recusive(arr):
        
        if len(arr) == 1:
            return arr
        mid = len(arr)>>1
        return merge(recusive(arr[:mid]), recusive(arr[mid:]))
     
    return recusive(arr)

merge_sort(arr)

def merge_sort(arr): # 对index进行操作，也就是在原arr上进行改动
    def merge(arr, l, mid, r):
        c = []
        i = l
        j = mid+1 
        while i <= mid and j <= r:
            if arr[i]>=arr[j]:
                c.append(arr[j])
                j += 1
            else:
                c.append(arr[i])
                i += 1
        c =  c + arr[i:mid+1] + arr[j:r+1]
        for i in range(l, r+1):
            arr[i] = c[i-l]
    
    def recusive(arr, l, r):
        
        if r-l == 0:
            return 
        mid = (l+r)>>1
        recusive(arr, l, mid)
        recusive(arr, mid+1,r)
        return merge(arr, l, mid, r)
     
    return recusive(arr, 0, len(arr)-1)




def xiaohe(arr): 小和问题
    
    # 这个是自己的思路 主要是b[j]作为对比，而视频中以a[i]作为对比，更加简洁
    def merge1(a, b):
        c = []
        i = j = 0
        ans = he = 0
        while i<len(a) and j<len(b):
            if a[i]<b[j]:
                he += a[i]
                c.append(a[i])
                i += 1
            else:
                c.append(b[j])
                ans += he
                j += 1
        c = c + a[i:] + b[j:]
        ans += he*(len(b[j:]))
        return c, ans
    
    def merge2(a, b):
        c = []
        i = j = 0
        ans = 0
        while i<len(a) and j<len(b):
            if a[i]<b[j]:
                ans += a[i]*len(b[j:])
                c.append(a[i])
                i += 1
            else:
                c.append(b[j])
                j += 1
        c = c + a[i:] + b[j:]
        return c, ans
    
    def recursive1(arr):
        
        if len(arr) == 1:
            return arr, 0
        mid = len(arr)>>1
        c1, sum1 = recursive1(arr[:mid])
        c2, sum2 = recursive1(arr[mid:])
        c3, sum3 = merge1(c1, c2)       
        return c3, sum3 + sum2 + sum1
    
    def recursive2(arr):
        if len(arr) == 1:
            return arr, 0
        mid = len(arr)>>1
        c1, sum1 = recursive2(arr[:mid])
        c2, sum2 = recursive2(arr[mid:])
        c3, sum3 = merge2(c1, c2)
        return c3, sum3 + sum2 + sum1

    print(recursive1(arr))
    print(recursive2(arr))
  
  # 逆序对同理
  def nixuwenti(arr):
    
    def merge(a, b):
        c = []
        i = j = 0
        ans = []
        while i<len(a) and j<len(b):
            if a[i]<=b[j]:
                c.append(a[i])
                i += 1
            else:
                ans = ans + [(a[k1], b[j]) for k1 in range(i, len(a))]
                c.append(b[j])
                j += 1
        c = c + a[i:] + b[j:]
        return c, ans
    
    def recursive(arr):
        
        if len(arr) == 1:
            return arr, []
        mid = len(arr)>>1
        c1, sum1 = recursive(arr[:mid])
        c2, sum2 = recursive(arr[mid:])
        c3, sum3 = merge(c1, c2)       
        return c3, sum1 + sum2 + sum3
    
   
    print(recursive(arr))

 
  
  # 荷兰国旗思想下的快排
  def quicksort(arr, l, r):
    
    def swap(index1, index2):
        arr[index1], arr[index2] = arr[index2], arr[index1]
    
    def partition(l, r):
        # 随机快排：改善结构
        # swap(l+random.randint(0, r-l), r)
        more = r
        less = l-1
        curr = l
        while curr < more:
            if arr[curr] > arr[r]:
                more -= 1
                swap(curr, more)
            elif arr[curr] < arr[r]:
                less += 1
                swap(curr, less)
                curr += 1
            else:
                curr += 1
        swap(more, r)
        return [less+1, more]
    
    if r <= l:
        return 
    less, more = partition(l, r)  
    quicksort(arr, l, less-1) 
    quicksort(arr, more+1, r)
  arr = return_random_arr() 

arr = return_random_arr() 
l, r = 0, len(arr)-1
quicksort(arr, 0, len(arr)-1)
print(arr)

# 堆排序

def heapsort(arr):  # 这里维持的是大根堆
    
    def swap(index1, index2):
        arr[index1], arr[index2] = arr[index2], arr[index1]
        
    def heapinsert(index): # 插入一个数字仍要保持堆结构
        while index:
            parent_index = (index-1)>>1
            if arr[index] > arr[parent_index]: #父节点小于子节点
                swap(index, parent_index)
                index = parent_index
            else:
                break
    
    def heapify(index, heapsize): # 变动其中一个数字，仍要保持堆结构
        left = 2*index+1
        while left < heapsize:
            largest = left+1 if (left+1 < heapsize) and (arr[left]<arr[left+1]) else left # 这里需要思考，根据判断条件巧妙解决了只有左子树的问题
            largest = largest if arr[index] < arr[largest] else index
            if largest == index:
                break
            swap(index, largest)
            index = largest
            left = 2*index+1
        #print(arr)
            
    heapsize = len(arr)
    if heapsize < 2:
        return 
    for i in range(len(arr)): # 首先是大根堆的排序
        heapinsert(i)
    #print(arr)
    #接下来就是头部弹出
    while heapsize>0:
        heapsize -= 1
        swap(0, heapsize) # 把小的值换上去
        #print(heapsize, arr)
        heapify(0, heapsize) # 再把小值沉下去
    
arr = return_random_arr() 
heapsort(arr)
check(arr)

# 根据大根堆 小根堆的思想 查找数据流的中位数
# 也可以使用现成的heapq模块，代码可以缩减很多的
def median_finder(arr):
    
    def swap(arr, index1, index2):
        arr[index1], arr[index2] = arr[index2], arr[index1]
        
    def heapinsert(arr, index, func):
        while index:
            parent_index = (index-1)>>1 # 细节
            if not func(arr[parent_index],arr[index]):
                swap(arr, parent_index, index)
                index = parent_index # 细节
            else:
                break
            
    def heapify(arr, index, heapsize, func):
        left = index*2 + 1
        while left<heapsize:
            change_index = left + 1 if left+1<heapsize and func(arr[left+1], arr[left]) else left # 细节
            change_index = index if func(arr[index], arr[change_index]) else change_index # 细节
            if change_index == index:
                break
            swap(arr, change_index, index)
            index = change_index
            left = index*2 + 1
            
        
    length = len(arr)
    if not arr:
        return 
    if length<2:
        return arr[0]
    maxheap = [] # 大根堆
    minheap = [] # 小根堆
    max_len = min_len = 1
    i,j = (0,1) if arr[0]>=arr[1] else (1,0)
    maxheap.append(arr[j])
    minheap.append(arr[i])
    fmax = lambda x1,x2: True if x1>=x2 else False
    fmin = lambda x1,x2: True if x1<=x2 else False
    
    def heap_add(heap, value, heapsize, func):
        try:
            heap[heapsize] = value
        except:
            heap.append(value)
        heapinsert(heap, heapsize, func)
        
    for i in range(2, length):
        if arr[i]<=maxheap[0] or (arr[i]>maxheap[0] and arr[i]<minheap[0] and max_len <= min_len):
            heap_add(maxheap, arr[i], max_len, fmax)
            max_len += 1
        elif arr[i]>=minheap[0] or (arr[i]>maxheap[0] and arr[i]<minheap[0] and max_len > min_len):
            heap_add(minheap, arr[i], min_len, fmin)
            min_len += 1
       
        # 调整两个堆的大小
        if max_len > min_len + 1:
            heap_add(minheap, maxheap[0], min_len, fmin)
            min_len += 1
            max_len -= 1
            swap(maxheap, 0, max_len)
            heapify(maxheap, 0, max_len, fmax)
        if min_len > max_len + 1:
            heap_add(maxheap, minheap[0], max_len, fmax) 
            max_len += 1
            min_len -= 1
            swap(minheap, 0, min_len)
            heapify(minheap, 0, min_len, fmin)
#        print(maxheap[:max_len], 'maxheap')
#        print(minheap[:min_len], 'minheap')
    if length%2:
        ans = maxheap[0] if max_len > min_len else minheap[0]
    else:
        ans = (maxheap[0] + minheap[0])/2
    return ans, maxheap[:max_len], minheap[:min_len]
            

arr = return_random_arr() 
ans, maxheap, minheap = median_finder(arr)
print(ans, maxheap, minheap)
x = sorted(arr)
length = len(x)
ans1 = (x[int(length/2)] + x[int(length/2)-1])/2 if not length%2 else x[length//2]
print(ans1==ans)    

def bucketsort(arr):
    # 桶排序的思想比较简单 浪费空间
    # 可以变成: 改进直接用字典，但是key的那部分其实也用到了比较
    if not arr or len(arr)<2:
        return
    import math
    max_num, min_num = -math.inf, math.inf
    for i in range(len(arr)):
        max_num = max(arr[i], max_num)
        min_num = min(arr[i], min_num)
    bucket = [0]*(max_num-min_num+1)
    for i in range(len(arr)):
        bucket[arr[i]-min_num] += 1
    l = 0
    for i, cnt in enumerate(bucket):
        if cnt>0:
            arr[l:l+cnt] = [min_num+i]*cnt
        l += cnt
arr = return_random_arr() 
bucketsort(arr)
check(arr)


# 利用桶排序的思想找到排序后的数组相邻最大的差值
# 多一个空桶的原因是为了证明至少空桶周围两个非空桶的差距肯定是大于桶内的差距的
def maxgap(arr):
    
    if not arr and len(arr)<2:
        return 
    import math
    min_num, max_num = math.inf, -math.inf
    for i in range(len(arr)):
        max_num = max(arr[i], max_num)
        min_num = min(arr[i], min_num)
    buckets_min = [math.inf]*(len(arr)+1) # 准备N+1个空桶
    buckets_max = [-math.inf]*(len(arr)+1)
    buckets_num = len(arr)
    for num in arr:
        index = int((num-min_num)/(max_num-min_num)*buckets_num)
        buckets_min[index] = min(num, buckets_min[index])
        buckets_max[index] = max(num, buckets_max[index])
    print(buckets_max, buckets_min)
    max_gap = 0
    for i in range(buckets_num+1):
        if buckets_min[i]>max_num:
            continue
        else:
            last_max = buckets_max[i]
            break
    for j in range(i+1, buckets_num+1):
        if buckets_min[j]>max_num:
            continue
        else: # 非空桶
            max_gap = max(max_gap, buckets_min[j]-last_max)
            last_max = buckets_max[j]
    return max_gap
arr = return_random_arr() 
ans1 = maxgap(arr)
x = sorted(arr)
ans2 = max([x[i+1]-x[i] for i in range(len(x)-1)])
print(ans1, ans2, ans1==ans2)




