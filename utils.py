import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from concurrent.futures import as_completed, ProcessPoolExecutor


import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from itertools import combinations


def convert_hyperedges_to_edges(hyperedges):
    graph_edges = set()
    for hyperedge in hyperedges:
        for pair in combinations(hyperedge, 2):
            graph_edges.add(pair)
    return [list(pair) for pair in graph_edges] #list(graph_edges)


def normalise(M):
    """
    row-normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  
    
    return DI.dot(M)


class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, reapproximate=True, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()
        


    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)



    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else: A = structure

        if self.cuda: A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'



class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2



def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators
    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without
    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for edge in E.tolist():
        hyperedge = list(edge)
        
        p = np.dot(X[hyperedge], rv)   #projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2*len(hyperedge) - 3    # normalisation constant
        if m:
            
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    
    
    return adjacency(edges, weights, V)



def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights



def adjacency(edges, n):
    """
    computes an sparse adjacency matrix
    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes
    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    # dictionary = {tuple(item): index for index, item in enumerate(edges)}
    # edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        # w = weights[(i,j)]
        organised.append(1)#(w)

    edges, weights = np.array(edges), np.array(organised)
    edges -= 1
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)
    # print(adj.shape)
    # A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    # A = ssm2tst(A)
    return adj



def symnormalise(M):
    """
    symmetrically normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 



def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)
    arguments:
    M: scipy sparse matrix
    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)
    


def add_padding_idx(vec):
	if len(vec.shape) == 1:
		return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
						 for v in tqdm(vec)])
	else:
		vec = np.asarray(vec) + 1
		vec = np.sort(vec, axis=-1)
		return vec.astype('int')


def np2tensor_hyper(vec, dtype):
	vec = np.asarray(vec)
	if len(vec.shape) == 1:
		return [torch.as_tensor(v, dtype=dtype) for v in vec]
	else:
		return torch.as_tensor(vec, dtype = dtype)


def walkpath2str(walk):
	return [list(map(str, w)) for w in tqdm(walk)]


def roc_auc_cuda(y_true, y_pred):
	try:
		y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
		y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
		return roc_auc_score(
			y_true, y_pred), average_precision_score(
			y_true, y_pred)
	except BaseException:
		return 0.0, 0.0


def accuracy(output, target):
	pred = output >= 0.5
	truth = target >= 0.5
	acc = torch.sum(pred.eq(truth))
	acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
	return acc


def build_hash(data):
	dict1 = set()

	for datum in data:
		# We need sort here to make sure the order is right
		datum.sort()
		dict1.add(tuple(datum))
	del data
	return dict1


def build_hash2(data):
	dict2 = set()
	for datum in tqdm(data):
		for x in datum:
			for y in datum:
				if x != y:
					dict2.add((x, y))
	return dict2


def build_hash3(data):
	dict2 = set()
	for datum in tqdm(data):
		for i in range(3):
			temp = np.copy(datum).astype('int')
			temp[i] = 0
			dict2.add(tuple(temp))

	return dict2


def parallel_build_hash(data, func, args, num, initial = None):
	import multiprocessing
	cpu_num = multiprocessing.cpu_count()
	data = np.array_split(data, cpu_num * 3)
	dict1 = initial.copy()
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	process_list = []

	if func == 'build_hash':
		func = build_hash
	if func == 'build_hash2':
		func = build_hash2
	if func == 'build_hash3':
		func = build_hash3

	for datum in data:
		process_list.append(pool.submit(func, datum))

	for p in as_completed(process_list):
		a = p.result()
		dict1.update(a)

	pool.shutdown(wait=True)
	
	return dict1

def generate_negative_edge(x, length):
	pos = np.random.choice(len(pos_edges), length)
	pos = pos_edges[pos]
	negative = []

	temp_num_list = np.array([0] + list(num_list))

	id_choices = np.array([[0, 1], [1, 2], [0, 2]])
	id = np.random.choice([0, 1, 2], length * neg_num, replace=True)
	id = id_choices[id]

	start_1 = temp_num_list[id[:, 0]]
	end_1 = temp_num_list[id[:, 0] + 1]

	start_2 = temp_num_list[id[:, 1]]
	end_2 = temp_num_list[id[:, 1] + 1]

	if len(num_list) == 3:
		for i in range(neg_num * length):
			temp = [
				np.random.randint(
					start_1[i],
					end_1[i]) + 1,
				np.random.randint(
					start_2[i],
					end_2[i]) + 1]
			while tuple(temp) in dict2:
				temp = [
					np.random.randint(
						start_1[i],
						end_1[i]) + 1,
					np.random.randint(
						start_2[i],
						end_2[i]) + 1]
			negative.append(temp)

	return list(pos), negative


def generate_outlier(k=20):
	inputs = []
	negs = []
	split_num = 4
	pool = ProcessPoolExecutor(max_workers=split_num)
	data = np.array_split(potential_outliers, split_num)
	dict_pair = build_hash2(np.concatenate([train_data, test]))

	process_list = []

	for datum in data:
		process_list.append(
			pool.submit(
				generate_outlier_part,
				datum,
				dict_pair,
				k))

	for p in as_completed(process_list):
		in_, ne = p.result()
		inputs.append(in_)
		negs.append(ne)
	inputs = np.concatenate(inputs, axis=0)
	negs = np.concatenate(negs, axis=0)

	index = np.arange(len(inputs))
	np.random.shuffle(index)
	inputs, negs = inputs[index], negs[index]

	pool.shutdown(wait=True)

	x = np2tensor_hyper(inputs, dtype=torch.long)
	x = pad_sequence(x, batch_first=True, padding_value=0).to(device)

	return (torch.tensor(x).to(device), torch.tensor(negs).to(device))

def pass_(x):
    return x


def generate_outlier_part(data, dict_pair, k=20):
	inputs = []
	negs = []
	
	for e in tqdm(data):
		point = int(np.where(e == 0)[0])
		start = 0 if point == 0 else int(num_list[point - 1])
		end = int(num_list[point])
		
		count = 0
		trial = 0
		while count < k:
			trial += 1
			if trial >= 100:
				break
			j = np.random.randint(start, end) + 1
			condition = [(j, n) in dict_pair for n in e]
			if np.sum(condition) > 0:
				continue
			else:
				temp = np.copy(e)
				temp[point] = j
				inputs.append(temp)
				negs.append(point)
				count += 1
	inputs, index = np.unique(inputs, axis=0, return_index=True)
	negs = np.array(negs)[index]
	return np.array(inputs), np.array(negs)


def check_outlier(model, data_):
	data, negs = data_
	bs = 1024
	num_of_batches = int(np.floor(data.shape[0] / bs)) + 1
	k = 3
	outlier_prec = torch.zeros(k).to(device)
	
	model.eval()
	with torch.no_grad():
		for i in tqdm(range(num_of_batches)):
			inputs = data[i * bs:(i + 1) * bs]
			neg = negs[i * bs:(i + 1) * bs]
			outlier = model(inputs, get_outlier=k)
			outlier_prec += (outlier.transpose(1, 0) == neg).sum(dim=1).float()
		outlier_prec = outlier_prec.cumsum(dim=0)
		outlier_prec /= data.shape[0]
		for kk in range(k):
			print("outlier top %d hitting: %.5f" % (kk + 1, outlier_prec[kk]))


class Word2Vec_Skipgram_Data_Empty(object):
	"""Word2Vec model (Skipgram)."""
	
	def __init__(self):
		return
	
	def next_batch(self):
		"""Train the model."""
		
		return 0, 0, 0, 0, 0
	