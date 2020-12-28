import time
import copy
import random


def binary_search(key: int, alist: list, start: int, end: int) -> int:
    print(start, end)
    if start > end:
        return -1
    mid = (start + end) // 2
    if alist[mid] > key:
        return binary_search(key, alist, start, mid-1)
    elif alist[mid] < key:
        return binary_search(key, alist, mid + 1, end)
    else:
        return mid


def max_common_divisor(a, b):
    if b == 0:
        return a
    mod = a % b
    return max_common_divisor(b, mod)


class UnionFind(object):
    """union-find 类"""

    def __init__(self, count):
        self.count = count
        self.value_list = []
        self.size_list = []
        for i in range(0, count):
            self.value_list.append(i)
            self.size_list.append(1)

    def union_count(self):
        return self.count

    def find(self, x: int):
        while self.value_list[x] != x:
            x = self.value_list[x]
        return x

    def union(self, a: int, b: int):
        a_boot = self.find(a)
        b_boot = self.find(b)
        if a_boot == b_boot:
            return
        if self.size_list[a_boot] >= self.size_list[b_boot]:
            self.value_list[b_boot] = a_boot
            self.size_list[a_boot] += self.size_list[b_boot]
        else:
            self.value_list[a_boot] = b_boot
            self.size_list[b_boot] += self.size_list[a_boot]
        self.count -= 1

    def connected(self, a: int, b: int):
        return self.find(a) == self.find(b)


class MultiSort(object):
    """多种排序方法的集合类"""
    ex_count = 0
    tem_list = []

    def exchange(self, a_list: list, a: int, b: int):
        _new = a_list[a]
        a_list[a] = a_list[b]
        a_list[b] = _new
        self.ex_count += 1

    def select_sort(self, a_list: list):
        total_count = len(a_list)
        for i in range(0, total_count):
            min_index = i
            j = i + 1
            while j < total_count:
                if a_list[j] < a_list[min_index]:
                    min_index = j
                j += 1
            self.exchange(a_list, i, min_index)

    def insert_sort(self, a_list: list):
        total_count = len(a_list)
        for i in range(1, total_count):
            j = i
            while j > 0 and a_list[j] < a_list[j-1]:
                self.exchange(a_list, j, j-1)
                j -= 1

    @staticmethod
    def tmp_insert_sort(a_list: list):
        total_count = len(a_list)
        for i in range(1, total_count):
            tmp = a_list[i]
            j = i - 1
            while j >= 0 and tmp < a_list[j]:
                a_list[j+1] = a_list[j]
                j -= 1
            a_list[j+1] = tmp

    def sher_sort(self, a_list: list):
        # 先确认h值
        total_count = len(a_list)
        h = 1
        while h < (total_count // 3):
            h = 3 * h + 1
        # 开始排序
        while h >= 1:
            for i in range(h, total_count):
                j = i
                while j >= h and a_list[j] < a_list[j-h]:
                    self.exchange(a_list, j, j - h)
                    j -= h
            h = h // 3

    def _merge(self, a_list: list, start: int, mid: int, end: int):
        i = start
        j = mid + 1
        # 先把原值复制到临时数组中
        for k in range(start, end+1):
            self.tem_list[k] = a_list[k]
        for k in range(start, end+1):
            if i > mid:
                a_list[k] = self.tem_list[j]
                j += 1
            elif j > end:
                a_list[k] = self.tem_list[i]
                i += 1
            elif self.tem_list[i] < self.tem_list[j]:
                a_list[k] = self.tem_list[i]
                i += 1
            else:
                a_list[k] = self.tem_list[j]
                j += 1

    def merge_sort(self, a_list: list, start: int, end: int):
        if start >= end:
            return
        mid = (start + end) // 2
        self.merge_sort(a_list, start, mid)
        self.merge_sort(a_list, mid+1, end)
        self._merge(a_list, start, mid, end)

    def merge_sort_2(self, a_list: list):
        total_count = len(a_list)
        sz = 1
        while sz < total_count:
            start = 0
            while start < (total_count - sz):
                self._merge(a_list, start, start+sz-1, min(start+sz+sz-1, total_count-1))
                start += 2 * sz
            sz += sz

    def merge_main(self, a_list: list):
        self.tem_list = copy.deepcopy(a_list)
        self.merge_sort_2(a_list)

    def split_list(self, a_list: list, start: int, end: int):
        tmp = a_list[start]
        i = start + 1
        j = end
        while True:
            while a_list[i] < tmp:
                i += 1
                if i > end:
                    break
            while a_list[j] > tmp:
                j -= 1
                # 可忽略，因为这里的tmp值取的就是start对应的值，不可能走到这一步
                if j < start:
                    break
            if i >= j:
                break
            self.exchange(a_list, i, j)
        # 最终交换一下对比值和j+1的位置
        self.exchange(a_list, start, j)
        return j

    def quick_sort(self, a_list: list, start: int, end: int):
        if end <= start:
            return
        j = self.split_list(a_list, start, end)
        self.quick_sort(a_list, start, j-1)
        self.quick_sort(a_list, j+1, end)

    def quick_sort_3way(self, a_list: list, start: int, end: int):
        if end <= start:
            return
        tmp = a_list[start]
        lt = start
        gt = end
        i = start + 1
        while i <= gt:
            if a_list[i] < tmp:
                self.exchange(a_list, lt, i)
                lt += 1
                # 这一步可以省略，这样代码更好看，因为大于等于的操作是对称的，加上的话会少一次等于情况的循环，其结果也是i+=1
                i += 1
            elif a_list[i] > tmp:
                self.exchange(a_list, i, gt)
                gt -= 1
            else:
                i += 1
        self.quick_sort_3way(a_list, start, lt-1)
        self.quick_sort_3way(a_list, gt+1, end)

    def quick_main(self, a_list: list):
        self.quick_sort_3way(a_list, 0, len(a_list)-1)


class PriorityQueue(object):

    def __init__(self):
        self.size = 0
        self.pq = [0]

    def is_empty(self):
        return self.size <= 0

    def exchange(self, i: int, j: int):
        temp = self.pq[i]
        self.pq[i] = self.pq[j]
        self.pq[j] = temp

    def swim(self, k: int):
        while k > 1 and self.pq[k] < self.pq[k//2]:
            self.exchange(k, k//2)
            k = k // 2

    def sink(self, k: int):
        while k <= self.size//2:
            j = 2 * k
            if j < self.size and self.pq[j] > self.pq[j+1]:
                j += 1
            if self.pq[j] >= self.pq[k]:
                break
            self.exchange(k, j)
            k = j

    def insert(self, a: int):
        self.pq.append(a)
        self.size += 1
        self.swim(self.size)

    def del_min(self):
        # 调换索引1和self.size的值
        min_value = self.pq[1]
        self.exchange(1, self.size)
        # 长度要减一
        self.size -= 1
        # 第一个元素下沉
        self.sink(1)
        return min_value


class IndexMInQueue(object):
    """索引优先队列"""

    def __init__(self, size):
        self.size = 0
        self.pq = []
        self.index_pqi = []
        self.pqi_index = []
        for i in range(0, size+1):
            self.index_pqi.append(-1)
            self.pqi_index.append(-1)
            self.pq.append(0)

    def is_empty(self):
        return self.size <= 0

    def exchange(self, i: int, j: int):
        temp = self.pq[i]
        self.pq[i] = self.pq[j]
        self.pq[j] = temp
        # 通过原本的值所在位置和索引关系表（pqi_index）找到对应索引
        k_i = self.pqi_index[i]
        k_j = self.pqi_index[j]
        # 索引和值所在位置关系表（index_pqi）更新
        self.index_pqi[k_i] = j
        self.index_pqi[k_j] = i
        # 反过来再更新--pqi_index
        self.pqi_index[j] = k_i
        self.pqi_index[i] = k_j

    def swim(self, k: int):
        while k > 1 and self.pq[k] < self.pq[k//2]:
            self.exchange(k, k//2)
            k = k // 2

    def sink(self, k: int):
        while k <= self.size//2:
            j = 2 * k
            # 2k或者2k+1中，较小值
            if j < self.size and self.pq[j] > self.pq[j+1]:
                j += 1
            # 如果较小值比k大，结束
            if self.pq[j] >= self.pq[k]:
                break
            # 如果较小值比k小，替换
            self.exchange(k, j)
            k = j

    def is_contains(self, k: int):
        return self.index_pqi[k] != -1

    def insert(self, k: int, val):
        self.size += 1
        self.pq[self.size] = val
        self.index_pqi[k] = self.size
        self.pqi_index[self.size] = k
        self.swim(self.size)

    def change(self, k: int, val):
        # 先找出索引对应的值所在位置
        a = self.index_pqi[k]
        # 在保存值的数组中，直接更新值
        self.pq[a] = val
        self.swim(a)
        self.sink(a)

    def del_min(self):
        min_value_index = self.pqi_index[1]
        # 调换索引1和self.size的值
        self.exchange(1, self.size)
        # 长度要减一
        self.size -= 1
        # 第一个元素下沉
        self.sink(1)
        self.index_pqi[min_value_index] = -1
        return min_value_index


class HeapSort(object):
    pq = [0]

    def __init__(self, a_list: list):
        for item in a_list:
            self.pq.append(item)

    def exchange(self, i: int, j: int):
        temp = self.pq[i]
        self.pq[i] = self.pq[j]
        self.pq[j] = temp

    def sink(self, k: int, size: int):
        while k <= size//2:
            j = 2 * k
            if j < size and self.pq[j] < self.pq[j+1]:
                j += 1
            if self.pq[j] <= self.pq[k]:
                break
            self.exchange(k, j)
            k = j

    def sort(self):
        size = len(self.pq) - 1
        k = size // 2
        while k >= 1:
            self.sink(k, size)
            k -= 1
        while size > 1:
            self.exchange(1, size)
            size -= 1
            self.sink(1, size)


def kendall_tau(a: list, b: list):
    # 先以a元素顺序为主，得出一个a元素的索引的列表
    a_index = {}
    for i in range(0, len(a)):
        a_index[a[i]] = i
    # 查出B列表中的元素对应在A中的索引
    b_index = []
    for item in b:
        if item not in a_index:
            continue
        b_index.append(a_index[item])
    print(b_index)
    # 接下来只需要算出b_index倒序有多少对即可
    count = 0
    for j in range(0, len(b_index)-1):
        for k in range(j+1, len(b_index)):
            if b_index[k] < b_index[j]:
                count += 1
    print(count)


class RedBlackTree(object):
    root = None  # 根节点初始为空

    class Node(object):
        key = ''
        val = ''
        left = None
        right = None
        size = 1  # 初始数量为0
        color = 'RED'  # 新节点默认是“红”节点

        def __init__(self, key, val, size, color='RED'):
            if key is None:
                raise ValueError('key值不能为空')
            self.key = key
            self.val = val
            if size and size > 0:
                self.size = size
            self.color = color

    def size(self, h):
        if h is None:
            return 0
        return self.size(h.left) + self.size(h.right) + 1

    @staticmethod
    def is_red(h):
        """判断节点是否是红节点"""
        if h is None:
            return False
        return h.color == 'RED'

    def left_rotate(self, h: Node):
        """节点的左旋转操作"""
        tmp = h.right
        h.right = tmp.left
        tmp.left = h
        # 保留原来颜色
        tmp.color = h.color
        # 旋转的颜色一定是红色
        h.color = 'RED'
        # 保留原来的size
        tmp.size = h.size
        # 更新h节点的size
        h.size = self.size(h.left) + self.size(h.right) + 1
        return tmp

    def right_totate(self, h: Node):
        """节点的有旋转操作"""
        tmp = h.left
        h.left = tmp.right
        tmp.right = h
        # 保留原来颜色
        tmp.color = h.color
        # 旋转的颜色一定是红色
        h.color = 'RED'
        # 保留原来的size
        tmp.size = h.size
        # 更新h节点的size
        h.size = self.size(h.left) + self.size(h.right) + 1
        return tmp

    @staticmethod
    def flip_color(h: Node):
        """翻转节点的子节点颜色和本身颜色"""
        h.left.color = 'BLACK'
        h.right.color = 'BLACK'
        h.color = 'RED'

    def put(self, key, val):
        self.root = self._put(self.root, key, val)
        self.root.color = 'BLACK'

    def _put(self, h, key, val):
        if h is None:
            return self.Node(key, val, 1, 'RED')
        if h.key > key:
            h.left = self._put(h.left, key, val)
        elif h.key < key:
            h.right = self._put(h.right, key, val)
        else:
            h.val = val
        # 平衡操作
        if not self.is_red(h.left) and self.is_red(h.right):
            h = self.left_rotate(h)
        if self.is_red(h.left) and self.is_red(h.left.left):
            h = self.right_totate(h)
        if self.is_red(h.left) and self.is_red(h.right):
            self.flip_color(h)

        h.size = self.size(h.left) + self.size(h.right) + 1
        return h

    def get(self, key):
        return self._get(self.root, key)

    def _get(self, h, key):
        if h is None:
            return None
        if h.key > key:
            return self._get(h.left, key)
        elif h.key < key:
            return self._get(h.right, key)
        else:
            return h.val

    def print_all(self):
        self._print_all(self.root)

    def _print_all(self, h):
        if h is None:
            return
        print(h.key + ':' + h.color)
        self._print_all(h.left)
        self._print_all(h.right)


class HashDict(object):

    def __init__(self, size: int):
        self.count = 0
        self.keys = []
        self.vals = []
        self.size = size
        for i in range(0, size):
            self.keys.append(None)
            self.vals.append(None)

    def hash(self, key):
        return key.__hash__() % self.size

    def resize(self, new_size):
        new_i = HashDict(new_size)
        # 把现有的所有key值插入到新的类中
        for i in range(0, self.size):
            if self.keys[i] is not None:
                new_i.put(self.keys[i], self.vals[i])
        # 替换值
        self.size = new_i.size
        self.keys = new_i.keys
        self.vals = new_i.vals

    def put(self, key, val):
        # 调整数组大小
        if self.count >= self.size // 2:
            self.resize(2 * self.size)
        i = self.hash(key)
        while self.keys[i] is not None:
            if self.keys[i] == key:
                self.vals[i] = val
                return
            i = (i + 1) % self.size
        self.keys[i] = key
        self.vals[i] = val
        self.count += 1

    def get(self, key):
        i = self.hash(key)
        while self.keys[i] is not None:
            if self.keys[i] == key:
                return self.vals[i]
            i = (i + 1) % self.size
        return None

    def delete(self, key):
        i = self.hash(key)
        while self.keys[i] is not None:
            if self.keys[i] == key:
                break
            i = (i + 1) % self.size
        if self.keys[i] is None:
            return
        self.keys[i] = None
        self.vals[i] = None
        # 处理后续
        i = (i + 1) % self.size
        while self.keys[i] is not None:
            tmp_key = self.keys[i]
            tmp_val = self.vals[i]
            self.keys[i] = None
            self.vals[i] = None
            # 因为Put操作会加1
            self.count -= 1
            self.put(tmp_key, tmp_val)
            i = (i + 1) % self.size
        self.count -= 1
        # 调整数组大小
        if 0< self.count <= self.size // 8:
            self.resize(self.size // 2)


def sort_run():
    print(max_common_divisor(56, 64))
    # alist = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
    # print(binary_search(39, alist, 0, len(alist)-1))
    # instance = UnionFind(10)
    # instance.union(4, 3)
    # instance.union(3, 8)
    # instance.union(6, 5)
    # instance.union(9, 4)
    # instance.union(2, 1)
    # instance.union(8, 9)
    # instance.union(5, 0)
    # instance.union(7, 2)
    # instance.union(6, 1)
    # instance.union(1, 0)
    # instance.union(6, 7)
    # print(instance.union_count())
    # print(instance.value_list)
    # print(instance.size_list)
    # 排序算法性能验证
    # alist = []
    # for i in range(0, 5000):
    #     alist.append(random.randint(1, 5000))
    # blist = copy.deepcopy(alist)
    # clist = copy.deepcopy(alist)
    # dlist = copy.deepcopy(alist)
    # elist = copy.deepcopy(alist)
    # instance = MultiSort()
    # print(time.time())
    # instance.insert_sort(elist)
    # print(time.time())
    # instance.tmp_insert_sort(dlist)
    # print(time.time())
    # instance.sher_sort(alist)
    # print(time.time())
    # instance.merge_main(blist)
    # print(time.time())
    # instance.quick_main(clist)
    # print(time.time())
    # 优先队列
    # pq = PriorityQueue()
    # pq.insert(10)
    # pq.insert(8)
    # pq.insert(9)
    # pq.insert(3)
    # pq.insert(5)
    # pq.insert(7)
    # pq.insert(13)
    # print(pq.size)
    # print(pq.pq)
    # for i in range(0, pq.size):
    #     print(pq.del_max())
    # pq.heap_sort([13, 8, 10, 3, 5, 7, 9])
    # 堆排序
    # hs = HeapSort(alist)
    # print(time.time())
    # hs.sort()
    # print(time.time())
    # 计算kendall tau距离
    # a = [0, 3, 1, 6, 2, 5, 4]
    # b = [1, 0, 3, 6, 4, 2, 5]
    # kendall_tau(a, b)
    # 测试红黑二叉树
    instance = RedBlackTree()
    instance.put('S', 19)
    instance.put('E', 5)
    instance.put('A', 1)
    instance.put('R', 18)
    instance.put('C', 3)
    instance.put('H', 8)
    instance.put('X', 24)
    instance.put('M', 13)
    instance.put('P', 16)
    instance.put('L', 12)
    instance.print_all()
    # 散列表
    instance = HashDict(15)
    instance.put('S', 19)
    instance.put('E', 5)
    instance.put('A', 1)
    instance.put('R', 18)
    instance.put('C', 3)
    instance.put('H', 8)
    instance.put('X', 24)
    instance.put('M', 13)
    instance.put('P', 16)
    instance.put('L', 12)
    print(instance.size)
    print(instance.keys)
    print(instance.vals)
    instance.delete('F')
    instance.delete('E')
    instance.delete('H')
    instance.delete('M')
    instance.delete('A')
    instance.delete('S')
    instance.delete('R')
    print(instance.size)
    print(instance.keys)
    print(instance.vals)


if __name__ == '__main__':
    index_minq = IndexMInQueue(10)
    for i in range(0, 10):
        value = (10 - i) * 10
        if index_minq.is_contains(i):
            index_minq.change(i, value)
        else:
            index_minq.insert(i, value)
    # 更新值
    if index_minq.is_contains(3):
        index_minq.change(3, 55)
    else:
        index_minq.insert(3, 55)

    if index_minq.is_contains(5):
        index_minq.change(5, 77)
    else:
        index_minq.insert(5, 77)

    if index_minq.is_contains(9):
        index_minq.change(9, 33)
    else:
        index_minq.insert(9, 33)
    # 取出最小
    min_index = index_minq.del_min()
    print(min_index)
    index_minq.insert(min_index, 66)
    a = '111'
