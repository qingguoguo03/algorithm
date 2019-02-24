


# 利用数组实现固定的栈与队列的结构

class ArraysStack(object):
    
    def __init__(self, arr_len):
        self.arr = [None]*arr_len
        self.size = 0
        self.arr_len = arr_len
    
    def push(self, num):
        if self.size < self.arr_len:
            self.arr[self.size] = num
            self.size += 1
        else:
            raise Exception('栈满，不能继续压栈')
            
    def pop(self): # 删除在这个值，size减1
        if self.size:
            self.size -= 1
            return self.arr[self.size]
        else:
            raise Exception('栈中没有数')
            
    def peek(self):
        if self.size:
            return self.arr[self.size-1]
        else:
            return None

stack = ArraysStack(5)
stack.push(5)
stack.push(4)
stack.push(3)
stack.push(2)
stack.push(1)
stack.push(0)
stack.pop()
stack.pop()
stack.push(1)
stack.push(0)
stack.peek()
        
class ArrayQueue(object):
    
    def __init__(self, arr_len):
        self.arr = [None]*arr_len
        self.size = 0
        self.start = 0
        self.end = 0
        self.arr_len = arr_len
    
    def push(self, num):
        if self.size < self.arr_len:
            self.arr[self.end] = num
            self.size += 1
            self.end = self.end+1 if self.end+1<self.arr_len else 0
        else:
            raise Exception('队列满，不能继续进入队列')
            
    def pop(self):
        if self.size:
            num = self.arr[self.start]
            self.start = self.start+1 if self.start+1<self.arr_len else 0
            self.size -= 1
            return num
        else:
            raise Exception('队列中没有数')
            
    def peek(self):
        if self.size:
            return self.arr[self.start]
        else:
            return None
        
queue = ArrayQueue(5)
queue.push(5)
queue.push(4)
queue.push(3)
queue.push(2)
queue.push(1)
queue.push(0)
queue.pop()
queue.pop()
queue.push(1)
queue.push(0)
queue.peek()       
 
# 得到栈中的最小值，准备两个栈
# 没找到现成的栈模块 但是下面这个队列实现的是后进先出 可以当做栈来实现
import queue
arr = queue.LifoQueue()
arr.put(5)
arr.get(block=False)


# 自己实现一个栈结构
class ArraysStack(object):
    
    def __init__(self):
        self.arr = []
        self.size = 0
        self.arr_len = 0
        
    def push(self, num):
        try:
            self.arr[self.size] = num
        except:
            self.arr.append(num)
        self.size += 1
        
            
    def pop(self): # 删除在这个值，size减1
        if self.size:
            self.size -= 1
            return self.arr[self.size]
        else:
            raise Exception('栈中没有数')
            
    def peek(self):
        if self.size:
            return self.arr[self.size-1]
        else:
            return None
stack = ArraysStack()
stack.push(5)
stack.push(4)
stack.push(3)
stack.push(2)
stack.push(1)
stack.push(0)
stack.pop()
stack.peek()

class GetMinStack(object):
    
    def __init__(self):
        self.stack = ArraysStack()
        self.data = ArraysStack()
        
    def push(self, num):
        self.stack.push(num)
        min_ = self.data.peek()
        if min_:
            self.data.push(min(min_, num))
        else:
            self.data.push(num)
    
    def pop(self):
        self.data.pop()
        return self.stack.pop()
        
    
    def peek(self):
        return self.stack.peek()
        
    def get_min(self):
        return self.data.peek()
    
stack = GetMinStack()
stack.push(5)
stack.push(4)
stack.push(3)
stack.push(2)
stack.push(1)
stack.push(0)
stack.get_min()
stack.pop()
stack.get_min()
stack.peek()  
stack.pop()
stack.pop()
stack.pop()
stack.get_min()

# 用栈实现队列，用队列实现栈
class TwoStacksQueue(object):
    
    def __init__(self):
        self.stack = ArraysStack(), ArraysStack()
        
    def push(self, num):
        if self.stack[1].peek() != None:
            while self.stack[1].peek() !=None:
                self.stack[0].push(self.stack[1].pop())
        self.stack[0].push(num)

    def pop(self):
        if self.stack[1].peek() is None:
            while self.stack[0].peek() != None:
                self.stack[1].push(self.stack[0].pop())
        return self.stack[1].pop()
        
    def peek(self):
        if self.stack[1].peek() is None:
            while self.stack[0].peek() != None:
                self.stack[1].push(self.stack[0].pop())
        return self.stack[1].peek()

q = TwoStacksQueue()
q.push(5)
q.push(4)
q.push(3)
q.push(2)
q.push(1)
print(q.pop())
print(q.pop())
q.push(1)
q.push(0)
print(q.peek())  

# 这个用现成的队列 不然之前固定的队列不大好用，中间长度不够要重新扩展空间
import queue
class TwoQueueStack(object):
    
    def __init__(self):
        self.queue = [queue.Queue(), queue.Queue()]
        
    def push(self, num):
        self.queue[0].put(num)
        
    def pop(self):
        while self.queue[0].qsize()>1:
            self.queue[1].put(self.queue[0].get(block=False))
        num = self.queue[0].get(block=False)
        self.queue[1], self.queue[0] = self.queue[0], self.queue[1]
        return num
    
    def peek(self):
        num = None
        while not self.queue[0].empty():
            num = self.queue[0].get(block=False)
            self.queue[1].put(num)
        self.queue[1], self.queue[0] = self.queue[0], self.queue[1]
        return num

stk = TwoQueueStack()
stk.push(5)
stk.push(4)
stk.push(3)
stk.push(2)
stk.push(1)
stk.push(0)
print(stk.pop())
print(stk.peek())


# 顺时针打印矩阵，但是没办法处理特殊情况
# 类似于 row,1 这样的， 需要单独处理
def printMatrixEdges(arr):  # 主要就是扣边界
    
    def print_box(a, d, r):
        
        if a == r: # 就只有一行
            for j in range(a, d+1):
                res.append(arr[a][j])
            return
        if d == a: # 表示只有一列:
            for i in range(a, r+1):
                res.append(arr[i][d])
            return
        for j in range(a, d): # 上边第一行
            res.append(arr[a][j])
        for i in range(a, r): # 右边第一列
            res.append(arr[i][d])
        for j in range(d, a, -1): # 下边第一列
            res.append(arr[r][j])
        for i in range(r, a, -1): # 左边第一列
            res.append(arr[i][a]) 
        
    if not arr:
        return []
    res = []
    r = len(arr)-1
    a, d = 0, len(arr[0])-1
    while a<=r and d>=0 and a<=d: # 细节，限制条件
        print_box(a, d, r)
        a += 1
        d -= 1
        r -= 1
    return res
    
arr = [[ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ], [ 9, 10, 11, 12 ],[ 13, 14, 15, 16 ]]
printMatrixEdges(arr) 
arr = [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]
printMatrixEdges(arr) 
arr = [[ 1, 2,], [ 4, 5], [ 7, 8]]
printMatrixEdges(arr) 
arr = [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ],[ 10, 11, 12 ]]
printMatrixEdges(arr)  
arr = [[ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ], [ 9, 10, 11, 12 ]]
printMatrixEdges(arr) 
arr = [[1,11],[2,12],[3,13],[4,14],[5,15],[6,16],[7,17],[8,18],[9,19],[10,20]]
arr = []
arr = [[]]
arr = [[5, 6, 7]]
arr = [[5]]
arr = [[5], [6]]
arr = [[5], [6], [7]]
arr = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]


# 技巧性的做法：
def printMatrixEdges(arr):
    res = []
    while arr:
        res.extend(arr[0])
        arr = list(zip(*arr[1:]))[::-1]
    return res

def printMatrixEdges(matrix):
    return matrix and list(matrix.pop(0)) + printMatrixEdges(list(zip(*matrix))[::-1])
printMatrixEdges(arr)

# leet上看到扣边界写的比较简洁的，也就是 a d r 用四个变量代替，会更加清楚
 def spiralOrder(self, matrix: 'List[List[int]]') -> 'List[int]':
        if matrix == []:
            return []
        result = []
        left, up = 0,0 # 右移 下移
        right =len(matrix[0])-1
        down = len(matrix)-1
        direct = 0 #0:right, 1:down, 2:left， 3:up
        while True:
            if direct == 0: #第一行打印完 下移一行
                for i in range(left, right+1):
                    result.append(matrix[up][i])
                up+=1
            if direct == 1: # 右边第一行打印完 ，左移
                for i in range(up, down+1):
                    result.append(matrix[i][right])
                right -= 1
            if direct == 2: # 下边第一行打印完，上移
                for i in range(right,left-1,-1):
                    result.append(matrix[down][i])
                down-=1
            if direct == 3: # 左边第一行打印完，右移
                for i in range(down,up-1,-1):
                    result.append(matrix[i][left])
                left+=1
            if up>down or left>right: # 边界条件
                return result
            direct = (direct+1)%4    


# 矩阵顺时针旋转90度 leet上也有比较有技巧的做法

def rotateMatrix(arr):
 
    arr[:] = [list(item) for item in list(zip(*arr[::-1]))]


arr = [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]
arr = [[ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ], [ 9, 10, 11, 12 ],[ 13, 14, 15, 16 ]]
rotateMatrix(arr)
print(arr)

# 扣边界做, 主要就是旋转赋值的问题

def rotateMatrix(arr):
    
    def change_box(left, right, r, l):
        for j in range(left, right):
            arr[left][j], arr[r-j][left], arr[r-left][l-j], arr[j][l-left] = arr[r-j][left], arr[r-left][l-j], arr[j][r-left], arr[left][j]
    
    left, right = 0, len(arr)-1
    r = l = right
    while left<right:
        change_box(left, right, r, l)
        left += 1
        right -= 1
rotateMatrix(arr)
print(arr)




