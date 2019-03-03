
# 二叉树的相关问题，遍历，平衡，打印之类的

# 二叉树先序遍历 中序遍历 后续遍历
# 递归版本 比较简单 但是浪费空间
def printPreOrderRecur(head):
    if not head:
        return
    print(head.val)
    printPreOrderRecur(head.left)
    printPreOrderRecur(head.right)

def printInOrderRecur(head):
    if not head:
        return
    printInOrderRecur(head.left)
    print(head.val)
    printInOrderRecur(head.right)
    
def printPosOrderRecur(head):
    if not head:
        return
    printPosOrderRecur(head.left)
    printPosOrderRecur(head.right)
    print(head.val)
    
# 非递归 自己压栈来做, 最有趣的时候后序遍历的思考
    
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
        
def printPreOrderStack1(head):
    # 先打印head.val 然后压栈右边 在压栈左边
    if not head:
        return
    stack = ArrayStack()
    stack.push(head)
    while stack.peek():
        head = stack.pop()
        print(head.val)
        if head.right:
            stack.push(head.right)
        if head.left:
            stack.push(head.left)
            
def printPreOrderStack(head):
    # 先打印head.val 然后压栈右边 在压栈左边
    if not head:
        return
    stack = ArrayStack()
    while stack.peek() or head:
        if not head:
            head = stack.pop()
        print(head.val)
        if head.right:
            stack.push(head.right)
        head = head.left
            

def printInOrderStack(head): # 还是要思考的！
    # 先打印head.val 然后压栈右边 在压栈左边
    if not head:
        return
    stack = ArrayStack()
    while stack.peek() or head:
        if not head:
            head = stack.pop()
            print(head.val)
            head = head.right
        else:
            stack.push(head)
            head = head.left            
            
def printPosOrderStack1(head): # 还是要思考的！
    # 必须用两个栈,直接按照后序遍历打印顺序压栈，然后打印出来的
    # 技巧，前序遍历是中左右，改成中右左，然后一个栈去接，打印出来的就是左右中 正好是后续遍历
    if not head:
        return
    stack1 = ArrayStack()
    stack2 = ArrayStack()
    stack1.push(head)
    while stack1.peek():
        head = stack1.pop()
        stack2.push(head)
        if head.left:
            stack1.push(head.left)
        if head.right:
            stack1.push(head.right)
    while stack2.peek():
        print(stack2.pop().val)
        
def printPosOrderStack2(head): # 还是要思考的！
    # 必须用两个栈,直接按照后序遍历打印顺序压栈，然后打印出来的
    # 技巧，前序遍历是中左右，改成中右左，然后一个栈去接，打印出来的就是左右中 正好是后续遍历
    if not head:
        return
    stack1 = ArrayStack() # 压中右左
    stack2 = ArrayStack() # 接受stack1的输出
    while stack1.peek() or head:
        if not head:
            head = stack1.pop()
        stack2.push(head)
        if head.left:
            stack1.push(head.left)
        head = head.right
        
    while stack2.peek():
        print(stack2.pop().val)
    
def printPosOrderStack(head): # 还是要思考的！这个方法比较有技巧性
    # 只用一个栈 那么要考虑两个头 按照输出格式压栈
    if not head:
        return
    stack = ArrayStack() # 压中右左
    stack.push(head)
    cur = head
    while stack.peek():
        c = stack.peek() # 表示父节点
        if c.left and cur != c.left and cur != c.right: # 表示当前指针是未处理过的
            stack.push(c.left) # 直到走到最左边
        elif c.right and cur != c.right: # 左边的走完了 走右边
            stack.push(c.right)
        else: # 走到叶节点
            print(stack.pop().val) # 从栈中扔掉这个节点
            cur = c # 表示这个节点已经走过了
            
class LeftRightNode: # 生成二叉树
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

head = [LeftRightNode(i) for i in [5,3,8,2,4,1,7,6,10,9,11]]       
for i in range(1, len(head)):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
    else:
        head[int((i-1)/2)].left = head[i]

# 先序遍历
printPreOrderRecur(head[0])
printInOrderRecur(head[0])
printPosOrderRecur(head[0])
printPreOrderStack(head[0])
printInOrderStack(head[0])
printPosOrderStack(head[0])

# 如何巧妙的打印二叉树，和视频有点不一样 视频是竖着打 这个横着打 横着打确实比较难扩充 层数多了可能会有点乱的
def printTree(head):
    if not head:
        return None
    # 先遍历获得最大的深度，算出满二叉树可以有多少节点 追加list的时候到了数量就可以继续了
    def get_max_len(head):
        if not head:
            return 0
        return 1 + max(get_max_len(head.left), get_max_len(head.right))
        
    max_len = get_max_len(head)
    if max_len==1: # 只有一个节点话 直接打印就好
        print(head.val)
        return
    nums = 2**(max_len)-1
    printList = [head]
    i = len(printList)
    while i<nums+1:
        parent = printList[(i-1)//2]
        if not parent:
            printList.extend([None, None])
        else:
            if parent.left:
                printList.append(parent.left)
            else:
                printList.append(None)
            if parent.right:
                printList.append(parent.right)
            else:
                printList.append(None)
        i += 2
    
    ss = ' '
    i = 0
    for n in range(1, max_len+1):
        pre = int(2**(n-1)-1)
        now = int(2**(n-1))
        zhanwei = ((2**max_len-1)*2)//(now+2)*ss
        print_row = [str(printList[i].val) if printList[pre] else '#']
        i += 1
        while now>1:
            print_row.append(str(printList[i].val) if printList[i] else '#' )
            i += 1
            now -= 1
        print(zhanwei+zhanwei.join(print_row))
head[-1].left = LeftRightNode(12)
head[-1].right = LeftRightNode(13)
printTree(head[0])
        
# 在一个二叉树中找到一个节点的后继节点
# 后继主要是指 中序遍历里面一个节点后面的下一个节点是后继节点
# 左子树的后继节点就是父节点 右子树的节点是父节点的节点
# 1. 如果一个节点有右子树，那么后继节点是右子树的最左节点
# 2. 如果一个节点没有左右子树，后继节点就是父节点
# 主要的重点就是最后一个的处理 所以加上了循环处理
class Node(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None

def getSucessNode(head):
    
    def getLeftMost(head):
        while head.left:
            head = head.left
        return head
    
    if not head:
        return None
    if head.right:
        return getLeftMost(head.right)
    else:
        parent = head.parent # 左子树的节点就是父节点
        while parent and parent.right == head:
            # 右子树的节点: 二叉树最右边的路径最为特殊也就是最后一个节点是没有后继的，细节
            head = parent
            parent = head.parent
        return parent
    
head = [Node(i) for i in [6,3,9,1,4,8,10,2,5,7]]
for i in range(1, 7):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
        head[i].parent = head[int((i-2)/2)]
    else:
        head[int((i-1)/2)].left = head[i]
        head[i].parent = head[int((i-1)/2)]
head[3].right = head[-3]
head[-3].parent = head[3]
head[4].right = head[-2]
head[-2].parent = head[4]
head[5].left = head[-1]
head[-1].parent = head[5]

h1 = head[0].left.left
print(getSucessNode(h1).val)
h1 = head[0].left.left.right
print(getSucessNode(h1).val)
h1 = head[0].left
print(getSucessNode(h1).val)
h1 = head[0].left.right
print(getSucessNode(h1).val)
h1 = head[0].left.right.right
print(getSucessNode(h1).val)
h1 = head[0].right.left.left
print(getSucessNode(h1).val)
h1 = head[0].right.left
print(getSucessNode(h1).val)
h1 = head[0].right
print(getSucessNode(h1).val)
h1 = head[0].right.right
print(getSucessNode(h1).val)


# 二叉树的序列化与反序列化
# 二叉树的序列化与反序列
# 对三种遍历进行操作 另外节点处没有左子树或者右子树  要有占位符
# 按层遍历
class LeftRightNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

head = [LeftRightNode(i) for i in [5,3,8,2,4,1,7,6,10,9,11]]       
for i in range(1, len(head)):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
    else:
        head[int((i-1)/2)].left = head[i]
        
# 递归版的序列化
def serialByPre(head):
    # '_' 下划线是为了后面好切分
    if not head: 
        return '#_' 
    res = str(head.val) + '_'
    res += serialByPre(head.left)
    res += serialByPre(head.right)
    return res

class Node(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None

        
def reconByPreString1(pre_str):
    # 反序列的话 用两个指针，一个表示走过 一个表当前的头，用栈来压
    stack = ArrayStack()
    values = [s for s in pre_str.split('_') if s]
    h = head = LeftRightNode(int(values[0]))
    stack.push(head)
    i = 1
    while stack.peek() and i<len(values):
        c = stack.peek()
        if values[i] != '#':
            if c.right != head and c.left != head: # 父节点两边子树都没走过
                tmp = LeftRightNode(int(values[i]))
                stack.push(tmp)
                c.left = tmp
                i += 1
            elif c.left == head: # 右子树没有走过
                tmp = LeftRightNode(int(values[i]))
                stack.push(tmp)
                c.right = tmp
                i += 1
            else: # 表示这个节点走过了
                head = stack.pop()
        else: # values[i] == '#'
            i += 1
            if values[i] == '#': # 叶节点
                head = stack.pop()
                i += 1
    return h

def reconByPreString2(pre_str):
    # 视频中用的是递归的方法，然后使用队列 先进先出进行反序列的
    import queue
    values = queue.Queue()
    {values.put(s) for s in pre_str.split('_') if s}
    
    def make_node(values):
        value = values.get(block=False)
        if value == '#':
            return None
        head = LeftRightNode(int(value))
        head.left = make_node(values)
        head.right = make_node(values)
        return head
    return make_node(values)


pre_str = serialByPre(head[0])
print(pre_str)
# 先序遍历用队列 注意公式么么
h = reconByPreString(pre_str)
pre_str = serialByPre(h)   
print(pre_str)        
        
# 在head末尾在添加一个node
head[-1].left = LeftRightNode(12)
head[-1].right = LeftRightNode(13)
pre_str = serialByPre(head[0])
print(pre_str)
# 先序遍历用队列 注意公式么么
h = reconByPreString(pre_str)
pre_str = serialByPre(h)   
print(pre_str) 


def serialByLevel2(head):
    # 用队列处理即可, 在这个之前应该获得最大的深度，循环截至点
    if not head:
        return None
    import queue
    values = queue.Queue()
    res = []
    def get_max_len(head):
        if not head:
            return 0
        return 1 + max(get_max_len(head.left), get_max_len(head.right))
    max_len = get_max_len(head)+1
    nums = 2**max_len-1
    values.put(head)
    while not values.empty() and len(res)<nums:
        head = values.get(block=False)
        if not head:
            res.append('#') # 去掉就是视频的层次划分
            #values.put(None)
            #values.put(None)
        else:
            res.append(str(head.val))
            if head.left:
                values.put(head.left)
            else:
                values.put(None)
            if head.right:
                values.put(head.right)
            else:
                values.put(None)
    return '_'.join(res)

def serialByLevel1(head):
    # 视频中的方法 和我理解的不一样，相当于#只占位有数字的前面
    if not head:
        return None
    res = str(head.val) +  '_'
    import queue
    values = queue.Queue()
    values.put(head)
    while not values.empty():
        head = values.get(block=False)
        if head.left:
            res += str(head.left.val) + '_'
            values.put(head.left)
        else:
            res += '#_'
        if head.right:
            res += str(head.right.val) + '_'
            values.put(head.right)
        else:
            res += '#_'
    return res
        
def reconByByLevel2(pre_str):
    # 也是用队列来接就好
    import queue
    q = queue.Queue()
    values = [s for s in pre_str.split('_') if s]
    h = LeftRightNode(int(values[0]))
    q.put(h)
    i = 1
    while not q.empty():
        head = q.get(block=False)
        if values[i] != '#':
            head.left = LeftRightNode(int(values[i]))
            q.put(head.left)
        i += 1
        if values[i] != '#':
            head.right = LeftRightNode(int(values[i]))
            q.put(head.right)
        i += 1
    return h
     
def reconByByLevel1(pre_str):
    # 视频也是用队列来接，但是代码看上去简洁，就是把产生node部分提出去了
    import queue
    q = queue.Queue()
    values = [s for s in pre_str.split('_') if s]
    h = LeftRightNode(int(values[0]))
    q.put(h)
    i = 1
    def generateNone(value):
        if value == '#':
            return None
        else:
            return LeftRightNode(int(values[i]))
        
    while not q.empty():
        head = q.get(block=False)
        head.left = generateNone(values[i])
        if head.left:
            q.put(head.left)
        i += 1
        head.right = generateNone(values[i])
        if head.right:
            q.put(head.right)
        i += 1
    return h            
    


pre_str = serialByPre(head[0])
print(pre_str)
# 先序遍历用队列 注意公式细节
h = reconByPreString(pre_str)
pre_str = serialByPre(h)   
print(pre_str)        
        
# 在head末尾在添加一个node
head[-1].left = LeftRightNode(12)
head[-1].right = LeftRightNode(13)
pre_str = serialByPre(head[0])
print(pre_str)
# 先序遍历用队列 注意公式细节
h = reconByPreString(pre_str)
pre_str = serialByPre(h)   
print(pre_str) 

# 按层进行序列化
pre_str = serialByLevel1(head[0])
print(pre_str)
pre_str = serialByLevel2(head[0])
print(pre_str)
# 层次反序列化
h = reconByByLevel2(pre_str) 
pre_str = serialByLevel2(h)  
print(pre_str) 
h = reconByByLevel1(pre_str) 
pre_str = serialByLevel1(h)  
print(pre_str) 


# 折纸问题的解决
# 耦合性问题 从中间开始往左右 两边折痕是相反的
# 每增加新的一层，就是 下上 下上 下上的增加 可以看做是二叉树的新一层
# 头结点是下 左子树是下 右子树是上 （耦合）
# 从上往下打印就是打印二叉树的中序遍历即可
def printFold(N):
    #方向其实可以直接用 up 或者 down 来做
    def printFoldByLevel(i, N, direction):
        if i>N: # 全部打印
            return
        printFoldByLevel(i+1, N, True)
        print('down' if direction else 'up')
        printFoldByLevel(i+1, N, False)
    return printFoldByLevel(1, N, True)
N = 4
printFold(N)


# 判断二叉树是否平衡
# 视频中这个二叉树只强调左子树右子树的高度差在1之内即可
# 这里返回的是元祖，如果别的语言不能返回多个的话，可以传入list来接受多个返回值
def isBalancedTree(head):

    def getHeight(head, level):
        if not head:
            return level, True
        leftH, lflag = getHeight(head.left, level+1)
        rightH, rflag = getHeight(head.right, level+1)
        if abs(leftH-rightH)>1:
            return level, False
        return max(leftH, rightH), True
    
    h, flag = getHeight(head, 1)
    return h-1 if flag else 0
    
head = [LeftRightNode(i) for i in [1,2,3,4,5,6,7,8]]       
for i in range(1, len(head)):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
    else:
        head[int((i-1)/2)].left = head[i]
isBalancedTree(head[0])
head[-3].left = LeftRightNode(10)
head[-3].right = LeftRightNode(11)
isBalancedTree(head[0])

# 判断一棵树是否是BST
def isBST(head):
    # 搜索二叉树 父节点的值要在左子树与右子树的中间,不存在重复的值
    # 这个是有问题的，要保证所有节点左子树所有数都要小于右边的所有数 节点值在中间的
    if not head:
        return True    
    if head.left and head.val<head.left.val:
        return False
    else:
        flag = isBST(head.left)

    if not flag:
        return False
    
    if head.right and head.val>head.right.val:
        return False
    else:
        return isBST(head.right)

    return isBST(head)

# 做法一直接就是中序遍历的递归版本 然后打印出来是升序顺序即可

# leet上也有一道题，但是可以有等号的存在，因此要修正一下
class Solution:
    # 写的不是很简洁 思路就是每次遍历的时候记录子树的最大值，最小值，最后传到上个节点的时候要根据左子树，右子树进行不同的赋值
    def isValidBST(self, root: TreeNode) -> bool:
        head = root
        if not head:
            return True
        def isbst(head):
            import math
            # 每棵树都保存左子树，右子树的最大值与最小值
            res = [-math.inf, math.inf, -math.inf, math.inf, True]
            if not res[-1]:
                return res
            if head.left and (head.val<=head.left.val): 
                res[-1] = False
            else:
                if head.left: # 存在左子树，返回左子树的最大值
                    tmp = isbst(head.left)
                    res[0], res[1], res[-1] = tmp[0], tmp[1], tmp[-1]
                else:
                    res[0] = max(head.val, res[0]) # 记录左子树的最大最小值
                    res[1] = min(head.val, res[1])

            if not res[-1]: # 左子树就不符合了
                return res

            if head.right and head.val>=head.right.val:
                res[-1] = False
            else:
                if head.right:
                    tmp = isbst(head.right)
                    res[2:] = tmp[2:]
                else:
                    res[2] = max(head.val, res[2]) # 记录右子树的最大最小值
                    res[3] = min(head.val, res[3])
            if not res[-1]:
                return res
            if  (head.right and res[3] <= head.val)  or (head.left and head.val <= res[0]): # 节点小于左子树的最大值 大于右子树的最小值
                # 存在等号的时候 这里就是比较细节的地方，只有存在左右子树，才会做的判断
                res[-1] = False
            else: # 对子树进行合并 找到子树的最大值与最小值
                res[0] = res[2] = max(res[0], res[2])
                res[1] = res[3] = min(res[1], res[3])
            return res

        return isbst(head)[-1]
head = [LeftRightNode(i) for i in [3,2,5,1,4]] #   
head = [LeftRightNode(i) for i in [4,2,8,1,3,5,9]]   
head = [LeftRightNode(i) for i in [1,1]] #
head = [LeftRightNode(i) for i in [2,1,3]] 
head = [LeftRightNode(i) for i in [0,-1]] 
for i in range(1, len(head)):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
    else:
        head[int((i-1)/2)].left = head[i]
head[-1].left = LeftRightNode(7)
head[-1].right = LeftRightNode(11)
isBST(head[0])


#这个方法还是比较有技巧性的 但是总体效率应该也没那么高
def isBST1(head):
    # 视频里面用的这个方法很有技巧，也没有介绍，仔细才能看懂，思想如下：
    # 1. 每次父节点的值挂在左子树的最右的节点下，然后左子树每个右节点都挂着父节点
    # 2. 当左子树为空时，就可以判断父节点与右节点的关系，然后将节点转为右节点（回到了父节点的父节点）
    # 3. 重新又开始挂左子树的右节点 此时已经挂着了父节点 则还原为null
    # 4. 父节点 赋值为 父节点的父节点， 右节点（原来父节点的父节点 转到右子树上 重新做循环)
    if not head:
        return True
    cur1 = head
    pre = None
    while cur1:
        cur2 = cur1.left
        if cur2:
            while cur2.right and cur2.right != cur1: # 说明还没有挂过父节点
                cur2 = cur2.right
            if not cur2.right: # 挂上父节点
                cur2.right = cur1
                cur1 = cur1.left
                continue # 先把每颗左子树的右节点都挂满父节点
            else: # 说明已经挂过 做过判断 还原
                cur2.right = None
        
        if pre and pre.val>cur1.val:
            return False
        pre = cur1 # 子节点
        cur1 = cur1.right # 父节点
        continue
    return True

head = [LeftRightNode(i) for i in [3,2,5,1,4]] #   
head = [LeftRightNode(i) for i in [4,2,8,1,3,5,9]]   
head = [LeftRightNode(i) for i in [1,1]] #
head = [LeftRightNode(i) for i in [2,1,3]] 
head = [LeftRightNode(i) for i in [0,-1]] 
for i in range(1, len(head)):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
    else:
        head[int((i-1)/2)].left = head[i]
head[-1].left = LeftRightNode(7)
head[-1].right = LeftRightNode(11)
isBST1(head[0])

# 判断一棵树是否是平衡二叉树
def isCBT1(head):
    # 判断一棵树是否是CBT：
    # 按照层次进行遍历，设置flag判断节点是否开启，开启之后，每个点不能包含叶子节点了
    if not head:
        return True
    leaf = False
    import queue
    q = queue.Queue()
    q.put(head)
    while not q.empty():
        head = q.get(block=False)
        if (leaf and (head.left or head.right)) or ((not head.left) and head.right):
            return False
        if head.left:
            q.put(head.left)
#        else:
#            leaf = True
        if head.right:
            q.put(head.right)
        else:
            leaf = True
    return True

# 再简化一点就是只管加入None 当节点变成None 后面的节点就不可以是None，否则就是False
def isCBT2(head):
    if not head:
        return True
    import queue
    q = queue.Queue()
    q.put(head)
    flag = False
    while not q.empty():
        head = q.get(block=False)
        if flag and head:
            return False
        if not head:
            flag = True
            continue
        q.put(head.left)
        q.put(head.right)
    return True
        
head = [LeftRightNode(i) for i in [1,2,3,4,5,6]]
for i in range(1, len(head)):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
    else:
        head[int((i-1)/2)].left = head[i]   
isCBT2(head[0])   

# 判断完全二叉树的叶子个数，思想:
# 1. 如果左子树的深度 等于 右子树的深度 是满二叉树
# 2. 如果左子树不是满二叉树 则右子树的节点数可以算出
# 3. 递归进去左子树 母子同样的问题
def getHeight(head):
    if not head:
        return True, 0
    h = head
    llen = rlen = 1
    while head.left:
        head = head.left
        llen += 1
    head = h
    while head.right:
        head = head.right
        rlen += 1
    if llen == rlen:
        return True, llen
    else:
        return False, llen
def completeTreeNums(head):
    nums = 0
    if not head:
        return nums
    
    #nums = 1 不用赋值1了 后面都要减掉的 直接就0就好
    flag, res = getHeight(head.left)
    
    if not flag:
        #nums += 2**(res-1)-1
        nums += 2**(res-1)
        nums += completeTreeNums(head.left)
    else:
        #nums += 2**res-1
        nums += 2**res
        nums += completeTreeNums(head.right)
    return nums

head = [LeftRightNode(i) for i in [1,2,3,4,5,6,7,8,9]]
for i in range(1, len(head)):
    if not(i % 2): 
        head[int((i-2)/2)].right = head[i]
    else:
        head[int((i-1)/2)].left = head[i]   
completeTreeNums(head[0])     


