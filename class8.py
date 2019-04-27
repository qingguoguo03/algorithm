# 这一节主要聚焦 暴力递归与动态规划 （从递归改为动态规划）


# 求n！的两种思路
def factorial(n):
    # 递归来做
    if n == 1:
        return 1
    return n*factorial(n-1)

def factorial2(n):
    #正常思路来做
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

n = 5
print(factorial(n))
print(factorial2(n))

# 递归思想解决汉诺塔问题

def move(start, end, helper, n):
    if  n == 1:
        print('move %d: %s->%s' % (n, start, end))
        return
    move(start, helper, end, n-1)
    print('move %d: %s->%s' % (n, start, end))
    move(helper, end, start, n-1)

move('a','b','c',3)

# leet爬楼梯问题
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1:
            return 1
        return self.climbStairs(n-1) + self.climbStairs(n-2)
# 上面这种方法会超时,改成动态规划来做：斐波那契数列的感觉
class Solution:
    def climbStairs(self, n: int) -> int:
        # b缓存未来的楼梯数量
        a = b = 1
        for i in range(n):
            a, b = b, a+b
        return a
    


# 打印子序列的所有子集 考虑该字符是否 0 1 进行操作
    
def printSets(s, res, i):
    if i+1 > len(s):
        return
    tmp = [item+s[i] for item in res] # 加上字符
    res.extend(tmp)
    printSets(s, res, i+1)
    return
    
res = ['']
s = 'abc'
printSets(s, res, 0)
print(res)


# leet上一道类似的题目，
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        def dfs(nums, i, res):
            if i == len(nums):
                return res
            res.extend([item+[nums[i]] for item in res]) # 主要这里别写成死循环
            dfs(nums, i+1, res) 
        dfs(nums, 0, res)
        return res

   
# 一个栈，利用递归实现逆序
class ArrayStack(object):
    
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

def get_last_num(stack):
    
    if stack.size == 1:
        print(stack.pop())
        return
    if stack.peek():  
       num = stack.pop()
       get_last_num(stack)
       stack.push(num)

stack = ArrayStack()
for i in range(1,8):
    stack.push(i)
while stack.peek():
    get_last_num(stack)
    

# 如果逆序存储下来的话

def get_last_num(stack):
    
    if stack.size==1:
        return stack.pop()
    if stack.peek():
        num = stack.pop()
        last = get_last_num(stack)
        stack.push(num)
    return last

def reverse(stack):
    
    if not stack.peek():
        return None
    
    last = get_last_num(stack)
    reverse(stack)
    stack.push(last)
       
stack = ArrayStack()
for i in range(1,8):
    stack.push(i)   
reverse(stack)
while stack.peek():
    print(stack.pop())

# 从左上角走到右下角，每一步只能向右或者向下。沿途经过的数字要累加起来。返回最小的路径和

