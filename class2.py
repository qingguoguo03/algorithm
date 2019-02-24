


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

# 反转单向 双向列表： 比较简单

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class DoubleNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        self.last = None
        

x = [ListNode(i) for i in range(5)]
for i in range(4):
    x[i].next = x[i+1]
head1 = x[0]

x = [DoubleNode(i) for i in range(5)]
for i in range(4):
    x[i].next = x[i+1]
for i in range(4, 0, -1):
    x[i].last = x[i-1]
head2 = x[0]  

def printListNode(head):
    while head:
        print(head.val)
        head = head.next
printListNode(head1)

def printDoubleNode(head):
    while head:
        print('now:', head.val)
        if head.last:
            print('last:', head.last.val)
        head = head.next
       
printDoubleNode(head2)

def reverseListNode(head):
    pre = None
    while head:
        tmp = head.next
        head.next = pre
        pre = head
        head = tmp
    return pre
printListNode(head1)
head1_ = reverseListNode(head1)
printListNode(head1_)   

def reverseDoubleNode(head):
    pre = None
    while head:
        tmp = head.next
        head.next = pre
        head.last = tmp
        pre = head
        head = tmp
    return pre
printDoubleNode(head2)
head2_ = reverseDoubleNode(head2)
printDoubleNode(head2_) 

# Z字打印数组：从宏观的角度分析打印路径
# 矩阵按照z进行打印输出
def printMatrixZigZag(arr):
    if not arr:
        return
    res = []
    a = b = (0, 0) # 两个指针都是从0开始
    rows = len(arr)
    cols = len(arr[0])
    res.append(arr[0][0])
    bool_flag = True # 从右边到左边
    while a[0] <= rows-1: # 超过说明已经走到了右下角
        a = (a[0], a[1]+1) if a[1] < cols-1 else (a[0]+1, a[1])
        b = (b[0]+1, b[1]) if b[0] < rows-1 else (b[0], b[1]+1)
        if bool_flag:
            i, j = a[0], a[1]
            while i<= b[0]:
                res.append(arr[i][j]) 
                i += 1
                j -= 1
            bool_flag = not bool_flag
        else:
            i, j = b[0], b[1]
            while i>= a[0]:
                res.append(arr[i][j])
                i -= 1
                j += 1
            bool_flag = not bool_flag
    return res

arr = [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ]]
arr = [[ 1, 2, 3 ], [ 4, 5, 6 ]]
arr = [[ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ], [ 10, 11, 12]]
arr = [[1,11],[2,12],[3,13],[4,14],[5,15],[6,16],[7,17],[8,18],[9,19],[10,20]]
arr = [[5, 6, 7]]
arr = [[5]]
arr = [[5], [6]]
arr = [[5], [6], [7]]
arr = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
res = printMatrixZigZag(arr)
print(res)


# 在行列都排好序的矩阵中找数 , 利用排好序的规律，快速查找区域

def findNumInSortedMatrix(arr, num):
    
    if not arr:
        return False
    r, l = len(arr)-1, len(arr[0])-1
    i, j = 0, l # 初始要么在第一条边 要么在左边第一条边
    while (i <= r) and (j>=0) :
        if arr[i][j]>num:
            j -= 1
        elif arr[i][j] == num:
            return True
        else:
            i += 1 # 已经在边界了 往下走
    return False

arr =[[ 0, 1, 2, 3, 4, 5, 6 ],
    [10, 12, 13, 15, 16, 17, 18 ],
    [23, 24, 25, 26, 27, 28, 29 ],
    [44, 45, 46, 47, 48, 49, 50 ],
    [65, 66, 67, 68, 69, 70, 71 ],
    [96, 97, 98, 99, 100, 111, 122 ],
    [166, 176, 186, 187, 190, 195, 200 ],
    [233, 243, 321, 341, 356, 370, 380 ]]
num = 389
findNumInSortedMatrix(arr, num)



# 打印两个有序列表的公共部分，视频中的公共部分强调是共同的值
x = [ListNode(i) for i in range(5)]
for i in range(4):
    x[i].next = x[i+1]
head1 = x[0]
printListNode(head1)

x = [ListNode(i) for i in range(3,8)]
for i in range(4):
    x[i].next = x[i+1]
head2 = x[0]
printListNode(head2)

# 上面的公共部分就是[3,4]
def printCommonPart(head1, head2):
    
    while (head1 and head2):
        if head1.val == head2.val:
            print(head1.val)
            head1 = head1.next
            head2 = head2.next
        elif head1.val > head2.val:
            head2 = head2.next
        else:
            head1 = head1.next
printCommonPart(head1, head2)  

# 判断一个链表是否是回文结构, leetcode原题, 三种策略
class Solution:
    def isPalindrome1(self, head: ListNode) -> bool:
        # 方法一:栈的结构: 空间n
        if not head:
            return True
        tmp = []
        cur = head
        while cur:
            tmp.append(cur.val)
            cur =cur.next
        i = -1
        while head:
            if head.val != tmp[i]:
                return False
            i -= 1
            head = head.next
        return True
    
    def isPalindrome2(self, head: ListNode) -> bool:
        # 方法二:切分到中间，空间少一半
        if not head or not head.next: # 没有数 或者 一个数
            return True
        head1 = head
        head2 = head.next 
        while head2.next and head2.next.next: # 相当于走了一半
            head1 = head1.next # 每次跳一个
            head2 = head2.next.next # 每次跳两个
        tmp = []
        cur = head1
        while cur:
            tmp.append(cur.val)
            cur =cur.next
        i = -1
        n = -len(tmp)
        while i>=n:
            if head.val != tmp[i]:
                return False
            i -= 1
            head = head.next
        return True
    
    def isPalindrome3(self, head: ListNode) -> bool:
        # 只用1个空间 改变链表指向 然后再改回来
        if not head or not head.next:
            return True
        head1 = head
        head2 = head.next
        while head2.next and head2.next.next:
            head1 = head1.next
            head2 = head2.next.next
        flag = False # 加了一段判断是否是奇偶的问题， 如果不判断拼接会有问题的啊？ 不知道为啥视频里面没有说道这个
        if head2.next:
            flag = True
            head1 = head1.next.next
        else:
            head1 = head1.next
        # 对后半部分的链表进行反转
        pre = None
        while head1:
            tmp = head1.next
            head1.next = pre
            pre = head1
            head1 = tmp
        head1 = head3 = pre # 要把这个节点保存下来 后续还原
        while head1:
            if head1.val != head.val:
                return False
            head1 = head1.next
            head = head.next
        pre = None
        head1 = head3
        while head1:
            tmp = head1.next
            head1.next = pre
            pre = head1
            head1 = tmp
        if flag:
            head = head.next
            head.next = head1
        return True
        


  # 单向链表 按照某个值分为 大于 等于 小于 部分
def smallEqualBig1(head, pivot):
    
    # 第一种方法：视频介绍 感觉不是很好，浪费空间，也要做排序
    if not head:
        return head
    nodes = []
    while head:
        nodes.append(head)
        head = head.next
    # 荷兰国旗问题
    small, big = -1, len(nodes)
    i = 0
    while i<big:
        if nodes[i].val < pivot:
            small += 1
            nodes[i], nodes[small] = nodes[small], nodes[i]
            i += 1
        elif nodes[i].val > pivot:
            big -= 1
            nodes[i], nodes[big] = nodes[big], nodes[i]
        else:
            i += 1
    for i in range(len(nodes)-1):
        nodes[i].next = nodes[i+1]
    nodes[-1].next = None
    return nodes[0]

def smallEqualBig2(head, pivot):
    
    # 第二种方法：主要是切分成几个指针，最后在粘在一起来做
    if not head:
        return head
    sh = st = None
    bh = bt = None
    eh = et = None
    while head:
        tmp = head.next
        head.next = None
        if head.val < pivot:
            if not sh:
                sh = head
                st = head
            else:
                st.next = head
                st = head
        elif head.val == pivot:
            if not eh:
                eh = head
                et = head
            else:
                et.next = head
                et = head
        else:
            if not bh:
                bh = head
                bt = head
            else:
                bt.next = head
                bt = head
        head = tmp
    if st: # 细节在如何拼接  下面很聪明的做法
        st.next = eh
        et = et if et else st
    if et:
        et.next = bh
    return sh if sh else eh if eh else bh

x = [ListNode(i) for i in [7,9,1,8,5,2,5,4,3]]
for i in range(len([7,9,1,8,5,2,5,4,3])-1):
    x[i].next = x[i+1]
head = x[0]
printListNode(head)
head2 = smallEqualBig1(head, 5) # 因为用了快排不能保证稳定


class Solution:
    # 方法一 在于先收集 在拷贝关系
    def copyRandomList1(self, head: 'Node') -> 'Node':
        head_bag = {}
        if not head:
            return None
        h = head
        while head:
            node = Node(head.val, None, None)
            head_bag[head] = node
            head = head.next
        head = h
        while head:
            head_bag[head].next = head_bag.get(head.next, None)
            head_bag[head].random = head_bag.get(head.random, None)
            head = head.next
        return head_bag[h]
    
    # 方法二: 进阶：很有创意，就是结构copy一遍后解耦出来
    def copyRandomList(self, head: 'Node') -> 'Node':
        
        if not head:
            return None
        h = head
        while head:
            tmp = head.next
            node = Node(head.val, tmp, None)
            head.next = node
            head = tmp
        head = h
        while head:
            tmp = head.next.next
            head.next.random = head.random.next if head.random else None
            head = tmp
        r = res = h.next
        head = h
        while head:
            head.next = head.next.next if head.next else None
            res.next = res.next.next if res.next else None
            head = head.next
            res = res.next
        return r # 这里不能用h.next 注意结果已经修改了
    
    
# 两个单项链表找相交的位置 三种方法，最有意思的是第三种 利用环的思想，输出的时候记得还原链表结构
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode1(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        # 直接用字典保存遍历过的节点即可，但是这个的话就要N的空间
        if not headA or not headB:
            return None
        nodes = set()
        while headA or headB:
            if headA:
                if headA in nodes:
                    return headA
                else:
                    nodes.add(headA)
                headA = headA.next
            if headB:
                if headB in nodes:
                    return headB
                else:
                    nodes.add(headB)
                headB = headB.next
        return None
    
    def getIntersectionNode2(self, headA, headB):
        # 按照题目的要求只能开辟1个空间, 记录链表长度
        if not headA or not headB:
            return None
        a = headA
        b = headB
        lena = lenb = 0
        while a:
            lena += 1
            a = a.next
        while b:
            lenb += 1
            b = b.next
        if lenb > lena:
            a, b = headB, headA
        else:
            a, b = headA, headB
        for i in range(abs(lenb-lena)):
            a = a.next
        while a:
            if a != b:
                a = a.next
                b = b.next
            else:
                return a
        return None
            
            
    def getIntersectionNode3(self, headA, headB):
        # 利用环的思想，如果他们有交集，那么对一个链表做个环 下一个链表也就会有环了
        if not headA or not headB:
            return None
        a, b = headA, headB
        while a.next: # 把A弄成环
            a = a.next
        a.next = headA 
        
        p1 = p2 = b
        while p2.next and p2.next.next: # 看b是否存在环
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                p2 = b
                while p1 != p2:
                    p2 = p2.next
                    p1 = p1.next
                a.next = None
                return p1
        a.next = None
        return None
   

# 单链表相交的问题，考虑好单链表是否有环 其次单链表相交是什么情况等
# 都无环 回归到上面的解题
# 都有环：找到两个环的入口点，然后一个指针不动，另外一个走看是否走到了重复的位置之前碰到了指针

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
def printListNode(head):
    while head:
        print(head.val)
        head = head.next
        

def findFirstIntersectNode(head1, head2):
    
    def get_loop_node(head):
        if not head or (not head.next) or (not head.next.next):
            return None
        
        p1 = p2 = head
        while p2.next and p2.next.next:
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                p2 = head
                while p1 != p2:
                    p2 = p2.next
                    p1 = p1.next
                return p1
        return None


    def no_loop(head1, head2):
        if not head1 or (not head2):
            return None
        a = head1
        while a.next:
            a = a.next
        a.next = head1
        p = get_loop_node(head2)
        a.next = None
        if p:
            return p
        else:
            return None
        
    def both_loop(head1, head2, p1, p2):
        if p1 == p2: # 交点在环之前或者就是环
            tmp = p2.next
            p2.next = None
            p = no_loop(head1, head2)
            p2.next = tmp
            if p:
                return p
            else:
                None
        else:
            p = p2.next
            while p2 != p:
                if p != p1:
                    p = p.next
                else:
                    return p1
            return None

    
    if not head1 or (not head2):
        return None
    p1 = get_loop_node(head1)
    p2 = get_loop_node(head2)
    if (not p1) and (not p2):
        return no_loop(head1, head2)
    elif p1 and p2:
        return both_loop(head1, head2, p1, p2)
    else:
        return None
   
# 无环
head1 = [ListNode(i) for i in [1,2,3,4,5,6,7]]
for i in range(len([1,2,3,4,5,6,7])-1):
    head1[i].next = head1[i+1]
h1 = head1[0]
printListNode(h1)
# 无环, 连在h1的6 7 处
head2 = [ListNode(i) for i in [0,9,8]]
for i in range(len([0,9,8])-1):
    head2[i].next = head2[i+1]
head2[-1].next = head1[-2]
h2 = head2[0]
printListNode(h2)
printListNode(findFirstIntersectNode(h1, h2))

# 有环 4
head1 = [ListNode(i) for i in [1,2,3,4,5,6,7]]
for i in range(len([1,2,3,4,5,6,7])-1):
    head1[i].next = head1[i+1]
head1[-1].next = head1[3]
h1 = head1[0]


# 有环，连在h1的无环处 2
head2 = [ListNode(i) for i in [0,9,8]]
for i in range(len([0,9,8])-1):
    head2[i].next = head2[i+1]
head2[-1].next = head1[1]
h2 = head2[0]
print(findFirstIntersectNode(h1, h2).val)

# 有环，连在h1的环处 6
head2 = [ListNode(i) for i in [0,9,8]]
for i in range(len([0,9,8])-1):
    head2[i].next = head2[i+1]
head2[-1].next = head1[-2]
h2 = head2[0]
print(findFirstIntersectNode(h1, h2).val)


# 在一个先小后大的数组找到最小值，和leet上一道题很像 就是顺序数组在某个节点旋转了一下 然后在去查找值
# 大概意思就是大小大 或者小大 或者大小
def getLessIndex(arr):
    if not arr:
        return None
    if len(arr) == 1 or arr[0]<arr[1]:
        return 0
    if arr[-1] < arr[-2]:
        return len(arr)-1
    # 下面就只有大小大的关系了
    left, right = 1, len(arr)-2
    while left < right:
        mid = (left + right)>>1
        if arr[mid] > arr[mid-1]:
            right = mid-1
        elif arr[mid] > arr[mid+1]:
            left = mid+1
        else:
            return mid
    return left
arr = [ 6, 5, 3, 0, 4, 6, 7, 8 ]
getLessIndex(arr)




      



