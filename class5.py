
# 前缀树的思想，借用path,end统计
class TrieNode(object):
    def __init__(self, path=0, end=0):
        self.path = path
        self.end = end
        self.next = {}
    
    
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        """
        word = word.strip()
        if word:
            cur = self.root
            for w in word:
                if w not in cur.next:
                    cur.next[w] = TrieNode()
                cur.path += 1
                cur = cur.next[w]
            cur.end += 1
                    

    def delete(self, word):
        word = word.strip()
        # 首先应该是查找单词
        if word and self.search(word):
            cur = self.root
            for w in word:
                cur.next[w].path -= 1
                if cur.next[w].path == 0:
                    del cur.next[w] 
                    return
                cur = cur.next[w]
            cur.end -= 1
        
    def search(self, word):
        """
        Returns if the word is in the trie.
        """
        word = word.strip()
        if word:
            cur = self.root
            for w in word:
                if w not in cur.next:
                    return False
                cur = cur.next[w]
            return False if cur.end == 0 else True

    def startsWith(self, prefix): # 返回以这个开头的个数
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        word = prefix.strip()
        if word:
            cur = self.root
            for w in word:
                if w not in cur.next:
                    return False
                cur = cur.next[w]
            return cur.path
    

# Your Trie object will be instantiated and called as such:
trie = Trie();
print(trie.search('zuo'))
trie.insert('zuo')
print(trie.search('zuo'))
trie.delete('zuo')
print(trie.search('zuo'))
trie.insert('zuo')
trie.insert('zuo')
trie.delete('zuo')
print(trie.search('zuo'))
trie.delete('zuo')
print(trie.search('zuo'))
trie.insert('zuoa')
trie.insert('zuoac')
trie.insert('zuoab')
trie.insert('zuoad')
trie.delete('zuoa')
print(trie.search('zuoa'))
print(trie.startsWith('zuo'))




# 哈夫曼算最小成本 python中默认的是小根堆
import queue
def lessMoney(arr):
    q = queue.PriorityQueue()
    {q.put(item) for item in arr}
    ans = 0
    while q.qsize()>1:
        cur = q.get(block=False)+q.get(block=False)
        ans += cur
        q.put(cur)
    return ans
arr = [6,7,8,9]
arr = [10,20,30]
print(lessMoney(arr))



# 成本最小 收益最大的做法
# 对于成本来说应该采用小根堆
# 对于收益最大化应该采用大根堆
# leetcode上有到类似的题目叫做IPO
import heapq

class Node(object):
    def __init__(self, cost, profit):
        self.cost = cost
        self.profit = profit

class Heap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key= key
        if initial: # 表示列表转成堆
            self._data = [(key(item), i, item) for i,item in enumerate(initial)]
            heapq.heapify(self._data)
        else:
            self._data = initial if initial is not None else []
        
            
    def push(self, item):
        i = len(self._data)
        heapq.heappush(self._data, (self.key(item), i, item))
        i += 1
        
    def pop(self):
        return heapq.heappop(self._data)[-1]
        
    def empty(self):
        return True if not self._data else False
    
    def peek(self):
        return self._data[0][-1]
        
        

class Solution:
    def findMaximizedCapital(self, k, W, Profits, Capital):
        nodes = []
        for i in range(len(Capital)):
            nodes.append(Node(Capital[i], Profits[i]))
        costheap = Heap(initial=nodes, key=lambda x: x.cost) 
        profitheap = Heap(initial=[], key=lambda x: -x.profit)
        for i in range(k):
            while not costheap.empty() and costheap.peek().cost <= W:
                c = costheap.pop()
                print(c.cost, c.profit)
                profitheap.push(c)
            if profitheap.empty():
                return W
            cur = profitheap.pop()
            print(i, cur.cost, cur.profit)
            W += cur.profit
        return W

k=2
W=2
Profits=[1,2,3]
Capital=[11,12,13]
solution = Solution()
print(solution.findMaximizedCapital(k, W, Profits, Capital))
            
# 下面不再按照堆的思想直接排序来做: （如果要扣除成本后再加收益 下面这个方法没有上面堆的效率高）
def findMaximizedCapital(k, W, Profits, Capital):
    num = [x for x in zip(Profits, Capital)]
    num.sort() # 直接就是成本最小排序，收益最大排序
    # 找到第一个满足收益最大 但是成本并不超过的项目即可
    while k and num:
        i = len(num)-1
        while i>=0 and num[i][1]>W:
            i -= 1
        if i<0:
            return W
        W += num[i][0]
        num.pop(num[i]) # 这个操作完后就把这个pop出来 不继续操作
        k -= 1
    return W
   
