


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



