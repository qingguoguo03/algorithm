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

    
