
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
