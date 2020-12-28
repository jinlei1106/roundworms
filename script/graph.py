from script.algorithms import IndexMInQueue, PriorityQueue, UnionFind


class Node(object):
    """单链表的基础数据结构：节点"""

    def __init__(self, val):
        self.val = val
        self.link = None


class Bag(object):
    """背包类的实现"""

    def __init__(self):
        self.first = None

    def add(self, val):
        new_node = Node(val)
        if self.first is None:
            self.first = new_node
        else:
            new_node.link = self.first
            self.first = new_node

    def __iter__(self):
        self.current = self.first
        return self

    def __next__(self):
        ret = self.current
        if ret is None:
            raise StopIteration
        self.current = ret.link
        return ret.val


class Stack(object):
    """栈的实现"""

    def __init__(self):
        self.first = None

    def is_empty(self):
        return self.first is None

    def push(self, val):
        new_node = Node(val)
        if self.first is None:
            self.first = new_node
        else:
            new_node.link = self.first
            self.first = new_node

    def pop(self):
        """取出队列头"""
        ret = self.first
        self.first = ret.link
        ret.link = None
        return ret.val

    def __iter__(self):
        self.current = self.first
        return self

    def __next__(self):
        ret = self.current
        if ret is None:
            raise StopIteration
        self.current = ret.link
        return ret.val


class Queue(object):
    """先入先出队列的实现"""

    def __init__(self):
        self.first = None
        self.last = None

    def is_empty(self):
        return self.first is None

    def en_queue(self, val):
        new_node = Node(val)
        if self.is_empty():
            self.first = new_node
        else:
            # 加到首节点的后面
            self.last.link = new_node
        self.last = new_node

    def un_queue(self):
        """取出队列头"""
        ret = self.first
        self.first = ret.link
        ret.link = None
        return ret.val

    def __iter__(self):
        self.current = self.first
        return self

    def __next__(self):
        ret = self.current
        if ret is None:
            raise StopIteration
        self.current = ret.link
        return ret.val


class Edge(object):
    """加权边的数据结构"""

    def __init__(self, v, w, weight):
        self.v = v
        self.w = w
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __le__(self, other):
        return self.weight <= other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    def __ge__(self, other):
        return self.weight >= other.weight

    def either(self):
        return self.v

    def other(self, vertex):
        if vertex == self.v:
            return self.w
        elif vertex == self.w:
            return self.v
        else:
            raise ValueError('This edge not contains this vertex: %s' % vertex)


class DirectEdge(object):
    """有向加权边的数据结构"""

    def __init__(self, v, w, weight):
        self.v = v
        self.w = w
        self.weight = weight

    def fromm(self):
        return self.v

    def to(self):
        return self.w


class Graph(object):
    """无向图的基本数据结构"""

    def __init__(self, v: int):
        self.size = v
        self.sz = []
        for i in range(0, v):
            self.sz.append(Bag())

    def add_edge(self, i: int, j: int):
        self.sz[i].add(j)
        self.sz[j].add(i)

    def adj(self, v):
        return self.sz[v]


class DirectGraph(object):
    """有向图的基本数据结构"""

    def __init__(self, v: int):
        self.size = v
        self.sz = []
        for i in range(0, v):
            self.sz.append(Bag())

    def add_edge(self, i: int, j: int):
        self.sz[i].add(j)

    def adj(self, v):
        return self.sz[v]

    def reverse(self):
        new_dg = DirectGraph(self.size)
        for v in range(0, self.size):
            for w in self.adj(v):
                new_dg.add_edge(w, v)
        return new_dg


class WeightGraph(object):
    """加权无向图的基本数据结构"""

    def __init__(self, v: int):
        self.size = v
        self.sz = []
        for i in range(0, v):
            self.sz.append(Bag())

    def add_edge(self, e: Edge):
        v = e.either()
        w = e.other(v)
        self.sz[v].add(e)
        self.sz[w].add(e)

    def adj(self, v):
        return self.sz[v]

    def edges(self):
        b = Bag()
        for v in range(0, self.size):
            for e in self.adj(v):
                if e.other(v) > v:
                    b.add(e)
        return b


class DirectWeightGraph(object):
    """加权有向图的基本数据结构"""

    def __init__(self, size):
        self.size = size
        self.sz = []
        for i in range(0, self.size):
            self.sz.append(Bag())

    def add_edge(self, e: DirectEdge):
        self.sz[e.fromm()].add(e)

    def adj(self, v):
        return self.sz[v]

    def edges(self):
        b = Bag()
        for v in range(0, self.size):
            for e in self.adj(v):
                b.add(e)
        return b


class DeepFirstSearch(object):
    """深度优先算法"""

    def __init__(self, g: Graph, s: int):
        self.graph = g
        self.source = s
        self.marked = []
        self.path = []
        for i in range(0, self.graph.size):
            self.marked.append(False)
            self.path.append(i)

    def dfs(self):
        self._dfs(self.source)

    def _dfs(self, v):
        self.marked[v] = True
        for _node in self.graph.adj(v):
            if not self.marked[_node]:
                self.path[_node] = v
                self._dfs(_node)

    def is_combine(self, v):
        return self.marked[v]

    def path_to(self, v):
        if not self.is_combine(v):
            return 'no path'
        ps = str(v)
        while self.path[v] != self.source:
            ps += '-'
            ps += str(self.path[v])
            v = self.path[v]
        ps += '-'
        ps += str(self.source)
        return ps


class BreadFirstSearch(object):
    """广度优先算法"""

    def __init__(self, g: Graph, s: int):
        self.graph = g
        self.source = s
        self.marked = []
        self.path = []
        for i in range(0, self.graph.size):
            self.marked.append(False)
            self.path.append(i)
        self.queue = Queue()

    def bfs(self):
        self.queue.en_queue(self.source)
        self.marked[self.source] = True
        while not self.queue.is_empty():
            # 取出头元素
            v = self.queue.un_queue()
            # 所有与头元素相邻的节点，加入队列，并修改path
            for _node in self.graph.adj(v):
                if self.marked[_node]:
                    continue
                self.queue.en_queue(_node)
                self.marked[_node] = True
                self.path[_node] = v

    def is_combine(self, v):
        return self.marked[v]

    def path_to(self, v):
        if not self.is_combine(v):
            return 'no path'
        ps = str(v)
        while self.path[v] != self.source:
            ps += '-'
            ps += str(self.path[v])
            v = self.path[v]
        ps += '-'
        ps += str(self.source)
        return ps


class DeepFirstOrder(object):

    def __init__(self, g: DirectGraph):
        self.graph = g
        self.marked = []
        self.pre = Queue()
        self.post = Queue()
        self.reverse_post = Stack()
        for i in range(0, self.graph.size):
            self.marked.append(False)
        # 深度优先遍历，进行初始化--从0节点开始
        for v in range(0, self.graph.size):
            if not self.marked[v]:
                self.dfs(v)

    def dfs(self, v):
        self.pre.en_queue(v)
        self.marked[v] = True
        for w in self.graph.adj(v):
            if not self.marked[w]:
                self.dfs(w)
        self.post.en_queue(v)
        self.reverse_post.push(v)

    def pre_order(self):
        return self.pre

    def post_order(self):
        return self.post

    def reverse_post_order(self):
        return self.reverse_post


class SCC(object):

    def __init__(self, g: DirectGraph):
        self.marked = []
        self.id = []
        self.count = 0
        self.graph = g
        for i in range(0, self.graph.size):
            self.marked.append(False)
            self.id.append(0)
        g_order = DeepFirstOrder(g.reverse())
        for v in g_order.reverse_post_order():
            print(v)
            if not self.marked[v]:
                self.dfs(v)
                self.count += 1

    def dfs(self, v):
        self.marked[v] = True
        self.id[v] = self.count
        for w in self.graph.adj(v):
            if not self.marked[w]:
                self.dfs(w)

    def scc_count(self):
        return self.count

    def is_strong_connected(self, a, b):
        return self.id[a] == self.id[b]


class PrimMST(object):
    """Prim最小生成树"""

    def __init__(self, g: WeightGraph):
        self.edge_to = []
        self.dist_to = []
        self.marked = []
        for i in range(0, g.size):
            self.marked.append(False)
            self.dist_to.append(float('inf'))  # 无穷大
            self.edge_to.append(None)
        self.index_minq = IndexMInQueue(g.size)
        self.index_minq.insert(0, 0.0)
        while not self.index_minq.is_empty():
            min_index = self.index_minq.del_min()
            self.visitor(g, min_index)

    def visitor(self, g: WeightGraph, v):
        self.marked[v] = True
        for e in g.adj(v):
            w = e.other(v)
            if self.marked[w]:
                continue
            if e.weight < self.dist_to[w]:
                self.edge_to[w] = e
                self.dist_to[w] = e.weight
                if self.index_minq.is_contains(w):
                    self.index_minq.change(w, e.weight)
                else:
                    self.index_minq.insert(w, e.weight)

    def edges(self):
        return self.edge_to[1:]

    def weight(self):
        weight = 0.0
        for i in range(1, len(self.dist_to)):
            weight += self.dist_to[i]
        return weight


class KruskalMST(object):
    """Kruakal最小生成树"""

    def __init__(self, g: WeightGraph):
        self.mst = []  # 保存最小生成树的边
        pq = PriorityQueue()  # 生成一个保存所有边的优先队列
        for e in g.edges():
            pq.insert(e)
        # union-found数据，检测会不会成环
        uf = UnionFind(g.size)
        while not pq.is_empty() and len(self.mst) < (g.size - 1):
            # 取出最小边
            min_e = pq.del_min()
            v = min_e.either()
            w = min_e.other(v)
            # 判断当前边和现有边不会成环
            if uf.connected(v, w):
                continue
            uf.union(v, w)
            self.mst.append(min_e)

    def edges(self):
        return self.mst

    def weight(self):
        return sum([k.weight for k in self.mst])


class DijkstraSP(object):
    """Dijkstra最短路径--边的权重都为正"""

    def __init__(self, g: DirectWeightGraph, s):
        self.source = s
        self.edge_to = []
        self.dist_to = []
        for i in range(0, g.size):
            self.dist_to.append(float('inf'))  # 无穷大
            self.edge_to.append(None)
        self.index_minq = IndexMInQueue(g.size)
        self.dist_to[s] = 0
        self.index_minq.insert(s, 0.0)
        while not self.index_minq.is_empty():
            min_index = self.index_minq.del_min()
            self.relax(g, min_index)

    def relax(self, g: DirectWeightGraph, v):
        for e in g.adj(v):
            w = e.to()
            if (self.dist_to[v] + e.weight) < self.dist_to[w]:
                self.edge_to[w] = e
                self.dist_to[w] = self.dist_to[v] + e.weight
                if self.index_minq.is_contains(w):
                    self.index_minq.exchange(w, e.weight)
                else:
                    self.index_minq.insert(w, e.weight)

    def has_path_to(self, w):
        return self.dist_to[w] < float('inf')

    def dist_to(self, w):
        return self.dist_to[w]

    def path_to(self, w):
        if not self.has_path_to(w):
            return 'no path'
        ps = str(w)
        while self.edge_to[w].fromm() != self.source:
            ps += '-'
            ps += str(self.edge_to[w].fromm())
            w = self.edge_to[w].fromm()
        ps += '-'
        ps += str(self.source)
        return ps


class TopolSP(object):
    """
    基于拓扑排序的最短路径算法--要求是无环图
    拓扑和Dijkstra的唯一区别是，按照拓扑排序的顶点顺序进行放松操作即可
    所以不需要维护一个索引优先队列，且时间复杂度是线性的
    """

    def __init__(self, g: DirectWeightGraph, s):
        self.source = s
        self.edge_to = []
        self.dist_to = []
        for i in range(0, g.size):
            self.dist_to.append(float('inf'))  # 无穷大
            self.edge_to.append(None)
        self.index_minq = IndexMInQueue(g.size)
        self.dist_to[s] = 0
        # 按照拓扑的顺序，放松节点
        # 其实是有问题的，DeepFirstOrder初始化接收的是DirectGraph实例
        # 我们这里传入的是DirectWeightGraph实例
        # 修改一下DeepFirstOrder里面的代码即可支持，这里代码只做为样例，就直接这样写了
        order = DeepFirstOrder(g)
        for v in order.reverse_post_order():
            self.relax(g, v)

    def relax(self, g: DirectWeightGraph, v):
        for e in g.adj(v):
            w = e.to()
            if (self.dist_to[v] + e.weight) < self.dist_to[w]:
                self.edge_to[w] = e
                self.dist_to[w] = self.dist_to[v] + e.weight
                if self.index_minq.is_contains(w):
                    self.index_minq.exchange(w, e.weight)
                else:
                    self.index_minq.insert(w, e.weight)


class BellmanFordSP(object):
    """Bellman-Ford最短路径算法，支持一般图（含有环，含有负权重的边，甚至负权重的环）"""

    def __init__(self, g: DirectWeightGraph, s):
        self.cost = 0
        self.cycle = None  # 构成负权重环的所有边
        self.dist_to = []
        self.edge_to = []
        self.is_in = []
        self.queue = Queue()
        for i in range(0, g.size):
            self.dist_to.append(float('inf'))
            self.edge_to.append(None)
            self.is_in.append(False)
        self.dist_to[s] = 0
        self.is_in[s] = True
        self.queue.en_queue(s)
        while not self.queue.is_empty():
            v = self.queue.un_queue()
            self.is_in[v] = False
            self.relax(g, v)

    def relax(self, g: DirectWeightGraph, v):
        for e in g.adj(v):
            w = e.to()
            if (self.dist_to[v] + e.weight) < self.dist_to[w]:
                self.edge_to[w] = e
                self.dist_to[w] = self.dist_to[v] + e.weight
                if not self.is_in[w]:
                    self.queue.en_queue(w)
                    self.is_in[v] = True
            self.cost += 1
            if self.cost % g.size == 0:
                self.find_negative_cycle()

    def find_negative_cycle(self):
        # 具体实现是用self.edge_to里面非空的边，构成一个新图，去检测是否含有环且权重和是否为负数
        self.cycle = 1
        return

    def has_negative_cycle(self):
        return self.cycle is not None


def mian_1():
    # 先构建图
    g = Graph(6)
    g.add_edge(0, 5)
    g.add_edge(2, 4)
    g.add_edge(2, 3)
    g.add_edge(1, 2)
    g.add_edge(0, 1)
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(0, 2)
    # 初始化 深度优先查找的api, 起点设为0
    dfs = DeepFirstSearch(g, 0)
    dfs.dfs()
    # 检查
    print(dfs.path_to(5))
    # 初始化 广度优先查找的api, 起点设为0
    bfs = BreadFirstSearch(g, 0)
    bfs.bfs()
    # 检查
    print(bfs.path_to(5))


def main_2():
    # 先构建图
    g = DirectGraph(13)
    g.add_edge(2, 3)
    g.add_edge(0, 6)
    g.add_edge(0, 1)
    g.add_edge(2, 0)
    g.add_edge(11, 12)
    g.add_edge(9, 12)
    g.add_edge(9, 10)
    g.add_edge(9, 11)
    g.add_edge(3, 5)
    g.add_edge(8, 7)
    g.add_edge(5, 4)
    g.add_edge(0, 5)
    g.add_edge(6, 4)
    g.add_edge(6, 9)
    g.add_edge(7, 6)
    # 初始化 深度优先查找的api
    dfo = DeepFirstOrder(g)

    pre_str = ''
    for i in dfo.pre_order():
        pre_str += str(i)
        pre_str += ' '
    print(pre_str)

    post_str = ''
    for i in dfo.post_order():
        post_str += str(i)
        post_str += ' '
    print(post_str)

    reverse_post_str = ''
    for i in dfo.reverse_post_order():
        reverse_post_str += str(i)
        reverse_post_str += ' '
    print(reverse_post_str)
    # 逆序图
    r_g = g.reverse()
    r_dfo = DeepFirstOrder(r_g)

    pre_str = ''
    for i in r_dfo.pre_order():
        pre_str += str(i)
        pre_str += ' '
    print(pre_str)

    post_str = ''
    for i in r_dfo.post_order():
        post_str += str(i)
        post_str += ' '
    print(post_str)

    reverse_post_str = ''
    for i in r_dfo.reverse_post_order():
        reverse_post_str += str(i)
        reverse_post_str += ' '
    print(reverse_post_str)
    # 连通分量
    scc = SCC(g)
    print(scc.id)
    print(scc.scc_count())


def main_3():
    # 先构建图
    g = WeightGraph(8)
    g.add_edge(Edge(4, 5, 0.35))
    g.add_edge(Edge(4, 7, 0.37))
    g.add_edge(Edge(5, 7, 0.28))
    g.add_edge(Edge(0, 7, 0.16))
    g.add_edge(Edge(1, 5, 0.32))
    g.add_edge(Edge(0, 4, 0.38))
    g.add_edge(Edge(2, 3, 0.17))
    g.add_edge(Edge(1, 7, 0.19))
    g.add_edge(Edge(0, 2, 0.26))
    g.add_edge(Edge(1, 2, 0.36))
    g.add_edge(Edge(1, 3, 0.29))
    g.add_edge(Edge(2, 7, 0.34))
    g.add_edge(Edge(6, 2, 0.40))
    g.add_edge(Edge(3, 6, 0.52))
    g.add_edge(Edge(6, 0, 0.58))
    g.add_edge(Edge(6, 4, 0.93))
    prim_mst = PrimMST(g)
    print(prim_mst.weight())
    kruskal_mst = KruskalMST(g)
    print(kruskal_mst.weight())


if __name__ == '__main__':
    # 先构建图
    g = DirectWeightGraph(8)
    g.add_edge(DirectEdge(4, 5, 0.35))
    g.add_edge(DirectEdge(5, 4, 0.35))
    g.add_edge(DirectEdge(4, 7, 0.37))
    g.add_edge(DirectEdge(5, 7, 0.28))
    g.add_edge(DirectEdge(7, 5, 0.28))
    g.add_edge(DirectEdge(5, 1, 0.32))
    g.add_edge(DirectEdge(0, 4, 0.38))
    g.add_edge(DirectEdge(0, 2, 0.26))
    g.add_edge(DirectEdge(7, 3, 0.39))
    g.add_edge(DirectEdge(1, 3, 0.29))
    g.add_edge(DirectEdge(2, 7, 0.34))
    g.add_edge(DirectEdge(6, 2, 0.40))
    g.add_edge(DirectEdge(3, 6, 0.52))
    g.add_edge(DirectEdge(6, 0, 0.58))
    g.add_edge(DirectEdge(6, 4, 0.93))
    dijkstra_sp = DijkstraSP(g, 0)
    print(dijkstra_sp.path_to(1))
    print(dijkstra_sp.path_to(2))
    print(dijkstra_sp.path_to(3))
    print(dijkstra_sp.path_to(4))
    print(dijkstra_sp.path_to(5))
    print(dijkstra_sp.path_to(6))
    print(dijkstra_sp.path_to(7))
