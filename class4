
# 随机等概率从字典中获取单词 主要就是借助另外一个字典
# 删除单词时，要同步将index上移
import random
class RandomPool(object):
    
    def __init__(self):
        self.keyIndex = {}
        self.indexKey = {}
        self.size = 0
    
    def insert(self, key):
        if key not in self.keyIndex:
            self.size += 1
            self.keyIndex[key] = self.size
            self.indexKey[self.size] = key
            
    
    def delete(self, key):
        if key in self.keyIndex:
            index = self.keyIndex[key]
        # 将最后的那个移动上来
        last_key = self.indexKey[self.size]
        self.indexKey[index] = last_key
        self.keyIndex[last_key] = index
        del self.indexKey[self.size]
        self.size -= 1
        del self.keyIndex[key]
        
    def get_random(self):
        if self.size ==  0:
            return None
        randomIndex = random.randint(1, self.size)
        return self.indexKey[randomIndex]
        
p = RandomPool()
p.insert('apple')
p.insert('orange')
p.insert('banana')
print(p.get_random())
print(p.get_random())
print(p.get_random())

# 岛屿求解：1连在一起的算作一个岛，0表示水，计算总共有多少岛，下面是递归感染的方法

def isIslands(arr):
    # 递归进行传播 统计个数即可
    def infect(i, j, r, l):
        if (i>=r or j>=l or i<0 or j<0 or arr[i][j]!=1): # 感染条件 细节注意左右边界,判断不能等于1要放在最后
            return
        arr[i][j] = 2
        infect(i+1, j, r, l)
        infect(i-1, j, r, l)
        infect(i, j+1, r, l)
        infect(i, j-1, r, l)
    
    if (not arr) or (not arr[0]): # arr为空或者是arr[0]为空都不继续
        return None
    
    r = len(arr)
    l = len(arr[0])
    cnt = 0
    for i in range(r):
        for j in range(l):
            if arr[i][j] == 1:
                cnt += 1
                infect(i, j, r, l)
    return cnt
        

m1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 1, 1, 1, 0, 1, 1, 1, 0], 
		[0, 1, 1, 1, 0, 0, 0, 1, 0],
		[0, 1, 1, 0, 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 1, 1, 0, 0], 
		[0, 0, 0, 0, 1, 1, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0]]
print(isIslands(m1))
m2 = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 1, 1, 1, 1, 1, 1, 1, 0], 
		[0, 1, 1, 1, 0, 0, 0, 1, 0],
		[0, 1, 1, 0, 0, 0, 1, 1, 0], 
		[0, 0, 0, 0, 0, 1, 1, 0, 0], 
		[0, 0, 0, 0, 1, 1, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0]]
print(isIslands(m2))

# 补充：
# 那如果数据量特别大，考虑并行处理，那么难点就在合起来的时候边界如果合并
# 考虑并查集的思想，边界上如果两个点不是一个集合，那么减1，变成一个集合，之后再碰到就不不用减了，如果已经知道是一个集合不进行减法操作

# isislands用并查集来做：
class UF(object):
    
    def make_sets(self, arr):
        m, n = len(arr), len(arr[0])
        self.parent = [i*n+j for i in range(m) for j in range(n)] # 每个节点的父亲赋值为自己
        self.size = [1]*m*n
        self.count = sum([sum(map(int, item)) for item in arr])
        
    def find_head(self, i):
        p = self.parent[i]
        if p != i:
            p = self.find_head(p)
        self.parent[i] = p
        return p
    
    def union_heads(self, i, j):
        p1, p2 = self.find_head(i), self.find_head(j)
        if p1 != p2:
            p1, p2 = (p1, p2) if self.size[p1] >= self.size[p2] else (p2, p1)
            for i,index in enumerate(self.parent): # 全部父节点换成P1
                if index == p2:
                    self.parent[i] = p1
            self.size[p1] += self.size[p2]
            self.count -= 1
#            self.size[p2] = 1 # 这个赋值有没有应该都一样
       
#    def get_subset_nums(self):
#        return len(set(self.parent))
# 利用并查集的思想解决岛屿问题
def isIslands2(arr):
    # 并查集:
    if (not arr) or (not arr[0]):
        return None
    
    def go_flag(i, j, r, l):
        if i<0 or i>=r or j<0 or j>=l or arr[i][j] != 1:
            return False
        return True
    
    r = len(arr)
    l = len(arr[0])
    uf = UF()
    uf.make_sets(arr)
    direction = [(-1, 0),(1, 0),(0, 1),(0, -1)]
    for i in range(r):
        for j in range(l):
            if arr[i][j] == 1:
                for ii, jj in direction:
                    if go_flag(i+ii, j+jj, r, l):
                        uf.union_heads(l*i+j, l*(i+ii)+j+jj)
    return uf.count

# leet朋友圈的个数 并查集 与岛屿问题还是不同的
class UF(object):
    
    def make_sets(self, arr):
        n = len(arr)
        self.parent = [i for i in range(n)] # 每个节点的父亲赋值为自己
        self.size = [1]*n
        
    def find_head(self, i):
        p = self.parent[i]
        if p != i:
            p = self.find_head(p)
        self.parent[i] = p
        return p
    
    def union_heads(self, i, j):
        p1, p2 = self.find_head(i), self.find_head(j)
        if p1 != p2:
            p1, p2 = (p1, p2) if self.size[p1] >= self.size[p2] else (p2, p1)
            for i,index in enumerate(self.parent): # 全部父节点换成P1,一定要把p2的父节点换为p1 不然是错的！！！经验教训啊！！！！！！！！
                if index == p2:
                    self.parent[i] = p1
            self.size[p1] += self.size[p2]
#            self.size[p2] = 1 # 这个赋值有没有应该都一样
       
    def get_subset_nums(self):
        return len(set(self.parent))
    
class Solution(object):

    def findCircleNum(self, arr):
        if not arr or (not arr[0]):
            return 0
        uf = UF()
        uf.make_sets(arr)
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                if arr[i][j] == 1:
                    uf.union_heads(i, j)
        return uf.get_subset_nums()

arr = [[1,1,0],[1,1,0],[0,0,1]]
arr = [[1,1,0],[1,1,1],[0,1,1]]
arr = [[1,0,0,1],[0,1,1,0],[0,1,1,1],[1,0,1,1]]
arr = [[1,0,0,1],[0,1,1,0],[0,1,1,1],[1,0,1,1]]
if not arr or (not arr[0]):
    #return 0
    pass
uf = UF()
uf.make_sets(arr)
n = len(arr)
for i in range(n):
    for j in range(i+1, n):
        if arr[i][j] == 1:
            print('arr i,j:', i, j)
            uf.union_heads(i, j)
print(uf.get_subset_nums())
# 朋友圈还有一种做法比较喜欢 就是直接遍历 用个visited 表示这个朋友是否遍历过即可
# 然后数据集上似乎这种效率更加高
def findCircleNum(arr):
    if not arr or (not arr[0]):
        return 0
    from collections import deque
    n = len(arr)
    visited = set()
    q = deque()
    circle = 0
    for i in range(n):
        if i not in visited:
            circle += 1
            visited.add(j)
            q.append(j)
            while q:
                p1 = q.popleft()
                for p2, is_friend in enumerate(arr[p1]):
                    if is_friend and p2 not in visited:
                        q.append(p2)
                        visited.add(p2)
    return circle
print(findCircleNum(arr))




