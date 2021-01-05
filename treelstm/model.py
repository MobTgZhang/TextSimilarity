from treelstm import constants
import torch
import torch.nn as nn
class Tree(object):
    def __init__(self, ):
        self.parent = None
        self.num_children = 0
        self.children = list()
    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)
    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size
    def depth(self, ):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if (child_depth > count):
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
# 树状GRU
class ChildSumTreeGRU(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeGRU, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.rzx = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.rzh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.hx = nn.Linear(self.in_dim, self.mem_dim)
        self.hu = nn.Linear(self.mem_dim, self.mem_dim)
    def node_forward(self, inputs, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        rz = self.rzx(inputs) + self.rzh(child_h_sum)
        r, z = torch.split(rz, rz.size(1) // 2, dim=1)
        r, z = torch.sigmoid(r), torch.sigmoid(z)
        hp = torch.tanh(self.hx(inputs) + self.hu(torch.mul(r, child_h_sum)))
        ht = torch.mul(1 - z, child_h_sum) + torch.mul(z, hp)
        return ht

    def batch_forward(self, tree, inputs,outputs):
        num_children = len(tree.children)
        for idx in range(num_children):
            self.batch_forward(tree.children[idx], inputs,outputs)
        # 如果是叶子结点的话
        if (num_children == 0):
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_h = list(map(lambda x: x.state, tree.children))
            child_h = torch.cat(child_h, dim=0)
        tree.state = self.node_forward(inputs[tree.idx], child_h)
        return tree.state
    def forward(self,trees,inputs):
        h_states = []
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        o_states = torch.FloatTensor(batch_size,seq_len,self.mem_dim).fill_(0.0)
        for k in range(batch_size):
            h_state = self.batch_forward(trees[k], inputs[k],o_states[k])
            h_states.append(h_state)
        h_states = torch.cat(h_states, dim=0)
        return o_states,h_states
# 树状LSTM
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(u)
        f = torch.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c,h
    def batch_forward(self, tree, inputs):
        num_children = len(tree.children)
        for idx in range(num_children):
            self.batch_forward(tree.children[idx], inputs)

        if num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        def SetOputput(tree,output):
            num_child = len(tree.children)
            for k in range(num_child):
                SetOputput(tree.children[k],output)
            output[tree.idx] = tree.state[1]
        outputs = torch.FloatTensor(inputs.size(0),self.mem_dim).fill_(0)
        SetOputput(tree,outputs)
        return outputs,tree.state
    def forward(self,trees,inputs):
        c_states,h_states = [],[]
        batch_size = inputs.size(0)
        o_states = []
        for k in range(batch_size):
            o_state,(c_state,h_state) = self.batch_forward(trees[k], inputs[k])
            c_states.append(c_state)
            h_states.append(h_state)
            o_states.append(o_state.unsqueeze(dim=0))
        c_states = torch.cat(c_states,dim=0)
        h_states = torch.cat(h_states, dim=0)
        o_states = torch.cat(o_states,dim=0)
        return o_states,(c_states,h_states)
# 用于求文本的相似度的模型
class Similarity(nn.Module):
    def __init__(self, mem_dim, hid_dim):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hid_dim = hid_dim
        self.wh = nn.Linear(2 * self.mem_dim,self.hid_dim)
        self.wp = nn.Linear(self.hid_dim,1)
    def forward(self,lvec,rvec):
        mulmat = torch.mul(lvec,rvec)
        absmat = torch.abs(torch.add(lvec,-rvec))
        vec = torch.cat((mulmat,absmat),1)
        out = torch.sigmoid(self.wh(vec))
        pred = torch.sigmoid(self.wp(out))
        return pred
class SimilarityTreeLSTM(nn.Module):
    def __init__(self,args):
        super(SimilarityTreeLSTM, self,).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=constants.PAD, sparse=args.sparse)
        if args.freeze:
            self.embedding.weight.requires_grad = False
        self.childrensumtreelstm = ChildSumTreeLSTM(args.embedding_dim,args.memory_size)
        self.similarity = Similarity(args.memory_size,args.hidden_size)
    def forward(self,ltree,linputs,rtree,rinputs):
        linputs = self.embedding(linputs)
        rinputs = self.emn(rinputs)
        lstate, lhidden = self.childrensumtreelstm(ltree, linputs)
        rstate, rhidden = self.childrensumtreelstm(rtree, rinputs)
        output = self.similarity(lstate,rstate)
        return output
class SimilarityTreeGRU(nn.Module):
    def __init__(self,args):
        super(SimilarityTreeGRU, self,).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=constants.PAD, sparse=args.sparse)
        if args.freeze:
            self.embedding.weight.requires_grad = False
        self.childrensumtreegru = ChildSumTreeGRU(args.embedding_dim,args.memory_size)
        self.similarity = Similarity(args.memory_size,args.hidden_size)
    def forward(self,ltree,linputs,rtree,rinputs):
        linputs = self.embedding(linputs)
        rinputs = self.emn(rinputs)
        lstate, lhidden = self.childrensumtreelstm(ltree, linputs)
        rstate, rhidden = self.childrensumtreelstm(rtree, rinputs)
        output = self.similarity(lstate,rstate)
        return output
class SimilarityBiGRU(nn.Module):
    def __init__(self,args):
        super(SimilarityBiGRU, self,).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=constants.PAD, sparse=args.sparse)
        if args.freeze:
            self.embedding.weight.requires_grad = False
        self.bigru = nn.GRU(args.embedding_dim,args.memory_size,batch_first=True,bidirectional = True)
        self.similarity = Similarity(2*args.memory_size,args.hidden_size)
    def forward(self,linputs,rinputs):
        linputs = self.embedding(linputs)
        rinputs = self.emn(rinputs)
        lstate, lhidden = self.bigru(linputs)
        rstate, rhidden = self.bigru(rinputs)
        output = self.similarity(lstate,rstate)
        return output
class SimilarityBiLSTM(nn.Module):
    def __init__(self,args):
        super(SimilarityBiLSTM, self,).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=constants.PAD, sparse=args.sparse)
        if args.freeze:
            self.embedding.weight.requires_grad = False
        self.bilstm = nn.LSTM(args.embedding_dim,args.memory_size,batch_first=True,bidirectional = True)
        self.similarity = Similarity(2*args.memory_size,args.hidden_size)
    def forward(self,linputs,rinputs):
        linputs = self.embedding(linputs)
        rinputs = self.emn(rinputs)
        lstate, lhidden = self.bilstm(linputs)
        rstate, rhidden = self.bilstm(rinputs)
        output = self.similarity(lstate,rstate)
        return output

class SimilarityGRU(nn.Module):
    def __init__(self,args):
        super(SimilarityGRU, self,).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=constants.PAD, sparse=args.sparse)
        if args.freeze:
            self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(args.embedding_dim,args.memory_size,batch_first=True,bidirectional = False)
        self.similarity = Similarity(2*args.memory_size,args.hidden_size)
    def forward(self,linputs,rinputs):
        linputs = self.embedding(linputs)
        rinputs = self.emn(rinputs)
        lstate, lhidden = self.gru(linputs)
        rstate, rhidden = self.gru(rinputs)
        output = self.similarity(lstate,rstate)
        return output
class SimilarityLSTM(nn.Module):
    def __init__(self,args):
        super(SimilarityLSTM, self,).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=constants.PAD, sparse=args.sparse)
        if args.freeze:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(args.embedding_dim,args.memory_size,batch_first=True,bidirectional = True)
        self.similarity = Similarity(2*args.memory_size,args.hidden_size)
    def forward(self,linputs,rinputs):
        linputs = self.embedding(linputs)
        rinputs = self.emn(rinputs)
        lstate, lhidden = self.lstm(linputs)
        rstate, rhidden = self.lstm(rinputs)
        output = self.similarity(lstate,rstate)
        return output
class TestTreeLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(TestTreeLSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.testrnn = ChildSumTreeGRU(embedding_dim,hidden_dim)
    def forward(self,inputs,trees):
        inputs = self.embedding(inputs)
        o,h = self.testrnn(trees,inputs)
        print(o.size(),h.size())
