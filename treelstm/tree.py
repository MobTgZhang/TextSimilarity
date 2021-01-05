# ------------------------------------------------------------------------------
# Tree class for sentences.
# ------------------------------------------------------------------------------
class Tree(object):
    def __init__(self,idx):
        self.idx = idx
        self.parent = None
        self.children = list()
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
    def __str__(self):
        return str(self.idx)
    def __repr__(self):
        return str(self.idx)
    def __len__(self):
        def getlen(tree):
            numbers = 1
            if len(tree.children)>0:
                for item in tree.children:
                    numbers += getlen(item)
            return numbers
        return getlen(self)
    def preOrder(self):
        def preOrderVisit(tree):
            mystr = str(tree.idx)+","
            if len(tree.children)>0:
                for item in tree.children:
                    mystr += preOrderVisit(item)
            return mystr
        return preOrderVisit(self)
    def InOrder(self):
        def InOrderVisit(tree):
            mystr = ""
            if len(tree.children) > 0:
                for item in tree.children:
                    mystr += InOrderVisit(item)
            mystr += str(tree.idx)+","
            return mystr
        return InOrderVisit(self)
def createTree(sdp,idx):
    tree = Tree(idx-1)
    for item in sdp:
        if item[1] == idx:
            # 添加子节点
            child = createTree(sdp,item[0])
            tree.add_child(child)
    return tree