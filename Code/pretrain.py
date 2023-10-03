from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
import tensorflow.compat.v1 as tf

from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
from scipy.sparse import diags
import os
import time
import argparse
import warnings
from scipy import sparse
from numpy import inf

from Modules import *
from utils import *

import matplotlib as mpl
mpl.use("Agg")
import multiprocessing
from matplotlib import pyplot as plt
import random
import pickle

np.random.seed(0)
cpu_num = multiprocessing.cpu_count()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device_ids = [0, 1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--data', type=str, default='ramani')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('-i', '--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=2,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=0.25,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--dropout', type=float, default=0,
                        help='Return hyperparameter. Default is 1.')
    
    parser.add_argument('-d', '--diag', type=str, default='True',
                        help='Use the diag mask or not')
    parser.add_argument(
        '-f',
        '--feature',
        type=str,
        default='walk',
        help='Features used in the first step')
    parser.add_argument('--num-epoch', type=int, default=300,
                        help='Number of epochs. Default is 300.')

    parser.add_argument('--layers', default=2, type=int,
                        help='Layers of hyperGCN')
    parser.add_argument('--type', default='concat', type=str,
                        help='How to combine the outputs')
    parser.add_argument('--dense', default=0, type=int,
                        help='Dense connection of hyperGCN or no')
    parser.add_argument('--rezero', default=0, type=int,
                        help='Using rezero')
    parser.add_argument('--dropedge', default=1.0, type=float,
                        help='Percentage of edges to keep for each epoch')
    parser.add_argument('--data_split', default='original', type=str,
                        help='ratio of data split')
    parser.add_argument('--plot', default=False, type=bool,
                        help='plotting the auc on train and test')

    parser.add_argument('--input', default='original', type=str,
                        help='Types of input for Hyper-SAGNN')

    args = parser.parse_args()
    
    # args.model_name = 'model_{}_'.format(args.data)
    args.epoch = 25
    
    args.save_path = os.path.join(
        '../checkpoints/', args.data)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def train_batch_hyperedge(model, loss_func, batch_data, batch_weight, type, y=""):
    x = batch_data
    w = batch_weight

    # When label is not generated, prepare the data
    if len(y) == 0:
        x, y, w = generate_negative(x, "train_dict", type, w)
        index = torch.randperm(len(x))
        x, y, w = x[index], y[index], w[index]

    # forward
    pred = model(x, return_recon = False)

    loss = loss_func(pred, y, weight=w)
    return pred, y, loss


def train_epoch(args, model, loss_func, training_data, optimizer, batch_size, type):
    # Epoch operation in training phase
    # Simultaneously train on 2 models: hyperedge-prediction (1) & random-walk with skipgram (2)
    edges, edge_weight = training_data
    y = torch.tensor([])

    # Permutate all the data
    index = torch.randperm(len(edges))
    edges, edge_weight = edges[index], edge_weight[index]

    if len(y) > 0:
        y = y[index]

    # dropedge
    if args.dropedge < 1.0:
        G = model.node_embedding.G 
        rowcol = model.node_embedding.G._indices()

        val = model.node_embedding.G._values()
        idx = random.sample(range(len(val)), int(args.dropedge * len(val)))
        model.node_embedding.G = torch.sparse.FloatTensor(rowcol[:, idx], val[idx],
                                                     torch.Size(model.node_embedding.G.shape)).to(device) #+ identity_mtx

    model.train()
    
    bce_total_loss = 0
    acc_list, y_list, pred_list = [], [], []
    
    batch_num = int(math.floor(len(edges) / batch_size))
    bar = trange(batch_num, mininterval=0.1, desc='  - (Training) ', leave=False, )
    for i in bar:

        batch_edge = edges[i * batch_size:(i + 1) * batch_size]
        batch_edge_weight = edge_weight[i * batch_size:(i + 1) * batch_size]
        batch_y = ""
        if len(y) > 0:
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            if len(batch_y) == 0:
                continue
        pred, batch_y, loss = train_batch_hyperedge(model, loss_func, batch_edge, batch_edge_weight,
                                                                    type, y=batch_y)
        
        acc_list.append(accuracy(pred, batch_y))
        y_list.append(batch_y)
        pred_list.append(pred)
        
        for opt in optimizer:
            opt.zero_grad()

        # backward
        loss.backward()
        
        # update parameters
        for opt in optimizer:
            opt.step()
        
        bar.set_description(" - (Training) BCE:  %.4f" %
                            (bce_total_loss / (i + 1)))
        bce_total_loss += loss.item()
    y = torch.cat(y_list)
    pred = torch.cat(pred_list)
    auc1, auc2 = roc_auc_cuda(y, pred)

    if args.dropedge < 1.0:
        model.node_embedding.G = G
    return bce_total_loss / batch_num, np.mean(acc_list), auc1, auc2


def eval_epoch(model, loss_func, validation_data, batch_size, type, dict_name):
    ''' Epoch operation in evaluation phase '''
    bce_total_loss = 0
    
    model.eval()
    with torch.no_grad():
        validation_data, validation_weight = validation_data
        y = ""
        
        index = torch.randperm(len(validation_data))
        validation_data, validation_weight = validation_data[index], validation_weight[index]
        if len(y) > 0:
            y = y[index]
        
        pred, label = [], []
        
        for i in tqdm(range(int(math.floor(len(validation_data) / batch_size))),
                      mininterval=0.1, desc='  - (Validation)   ', leave=False):
            # prepare data
            batch_x = validation_data[i * batch_size:(i + 1) * batch_size]
            batch_w = validation_weight[i * batch_size:(i + 1) * batch_size]

            if len(y) == 0:
                batch_x, batch_y, batch_w = generate_negative(
                    batch_x, dict_name, type, weight=batch_w)
            else:
                batch_y = y[i * batch_size:(i + 1) * batch_size]
            
            index = torch.randperm(len(batch_x))
            batch_x, batch_y, batch_w = batch_x[index], batch_y[index], batch_w[index]
            
            pred_batch, _ = model(batch_x, return_recon = False)
            pred.append(pred_batch)
            label.append(batch_y)
            loss = loss_func(pred_batch, batch_y, weight=batch_w)
            
            bce_total_loss += loss.item()
        
        pred = torch.cat(pred, dim=0)
        label = torch.cat(label, dim=0)

        acc = accuracy(pred, label)
        auc1, auc2 = roc_auc_cuda(label, pred)

    return bce_total_loss / (i + 1), acc, auc1, auc2


def train(args, model, loss, training_data, validation_data, optimizer, epochs, batch_size, scheduler): # scheduler
    valid_accus = [0]
    loss_train = []
    loss_test = []

    for epoch_i in range(epochs):
        
        print('[ Epoch', epoch_i, 'of', epochs, ']')
        
        start = time.time()
        bce_loss, train_accu, auc1, auc2 = train_epoch(
            args, model, loss, training_data, optimizer, batch_size, train_type)
        print('  - (Training)   bce: {bce_loss: 7.4f}, '
              ' acc: {accu:3.3f} %, auc: {auc1:3.3f}, aupr: {auc2:3.3f}, '
              'elapse: {elapse:3.3f} s'.format(
            bce_loss=bce_loss,
            accu=100 *
                 train_accu,
            auc1=auc1,
            auc2=auc2,
            elapse=(time.time() - start)))
        
        start = time.time()
        valid_bce_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(args, model, loss, validation_data, batch_size,
                                                                                    'hyper', 'test_dict')
        print('  - (Validation-hyper) bce: {bce_loss: 7.4f},'
              '  acc: {accu:3.3f} %,'
              ' auc: {auc1:3.3f}, aupr: {auc2:3.3f},'
              'elapse: {elapse:3.3f} s'.format(
            bce_loss=valid_bce_loss,
            accu=100 *
                 valid_accu,
            auc1=valid_auc1,
            auc2=valid_auc2,
            elapse=(time.time() - start)))

        loss_train.append(auc1)
        loss_test.append(valid_auc1)

        scheduler.step()

        checkpoint = {
            'model_link': model.state_dict(),
            'epoch': epoch_i}

        model_name = 'model_conv.chkpt'
        
        if valid_auc1 >= max(valid_accus):
            torch.save(checkpoint, os.path.join(args.save_path, model_name))

        valid_accus += [valid_auc1]
        torch.cuda.empty_cache()
        
    checkpoint = torch.load(os.path.join(args.save_path, model_name))
    model.load_state_dict(checkpoint['model_link'])
    print("Saved at epoch # ", checkpoint['epoch'], file=open("result.txt", "a"))
    print("Max ROC AUC: ", max(valid_accus))

    valid_bce_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(args, model, loss, validation_data, batch_size,
                                                                                'hyper','test_dict')
    print('  - (Validation-hyper) bce: {bce_loss: 7.4f},'
          '  acc: {accu:3.3f} %,'
          ' auc: {auc1:3.3f}, aupr: {auc2:3.3f},'
          'elapse: {elapse:3.3f} s'.format(
          bce_loss=valid_bce_loss,
          accu=100 *
              valid_accu,
          auc1=valid_auc1,
          auc2=valid_auc2,
          elapse=(time.time() - start)), file=open("result.txt", "a"))

    # model.print_norm()
    if args.plot:
        plt.ylim(0.7,1)
        plt.plot(loss_train)
        plt.plot(loss_test)
        plt.title('model auc score')
        plt.ylabel('auc')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if args.dropedge == 0.95:
            plt.savefig('auc_train_test_95_' + str(args.data_split) + '.png')
        elif args.dropedge == 0.90:
            plt.savefig('auc_train_test_90_' + str(args.data_split) + '.png')
        elif args.dropedge == 0.80:
            plt.savefig('auc_train_test_80_' + str(args.data_split) + '.png')
        elif args.dropedge == 0.70:
            plt.savefig('auc_train_test_70_' + str(args.data_split) + '.png')
        elif args.dropedge == 0.60:
            plt.savefig('auc_train_test_60_' + str(args.data_split) + '.png')
        else:
            plt.savefig('auc_train_test_50_' + str(args.data_split) + '.png')

    
def test(args, model, loss, validation_data, batch_size):
    model_name = 'model_conv.chkpt'
    checkpoint = torch.load(os.path.join(args.save_path, model_name))
    model.load_state_dict(checkpoint['model_link'])

    valid_bce_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(args, model, loss, validation_data, batch_size,
                                                                                'hyper', 'test_dict')
    print('  - (test-hyper) bce: {bce_loss: 7.4f},'
          '  acc: {accu:3.3f} %,'
          ' auc: {auc1:3.3f}, aupr: {auc2:3.3f},'.format(
          bce_loss=valid_bce_loss,
          accu=100 *
              valid_accu,
          auc1=valid_auc1,
          auc2=valid_auc2), file=open("result.txt", "a"))


def generate_negative(x, dict1, get_type='all', weight="", forward=True):
    if dict1 == 'train_dict':
        dict1 = train_dict
    elif dict1 == 'test_dict':
        dict1 = test_dict
      
    if len(weight) == 0:
        weight = torch.ones(len(x), dtype=torch.float)
    
    neg_list = []
    
    zero_num_list = [0] + list(num_list)
    new_index = []
    max_id = int(num[-1])
    
    if forward:
        func1 = pass_
    else:
        func1 = tqdm
    
    if len(x.shape) > 1:
        change_list_all = np.random.randint(
            0, x.shape[-1], len(x) * neg_num).reshape((len(x), neg_num))
    for j, sample in enumerate(func1(x)):
        if len(x.shape) > 1:
            change_list = change_list_all[j, :]
        else:
            change_list = np.random.randint(0, sample.shape[-1], neg_num)
        for i in range(neg_num):
            temp = np.copy(sample)
            a = set()
            a.add(tuple(temp))
            
            trial = 0
            simple_or_hard = np.random.rand()
            if simple_or_hard <= pair_ratio:
                change = change_list[i]
                
            while not a.isdisjoint(dict1):
                temp = np.copy(sample)
                trial += 1
                if trial >= 1000:
                    temp = ""
                    break
                # Only change one node
                if simple_or_hard <= pair_ratio:
                    if len(num_list) == 1:
                        # Only one node type
                        temp[change] = np.random.randint(0, max_id, 1) + 1
                    
                    else:
                        # Multiple node types
                        start = zero_num_list[node_type_mapping[change]]
                        end = zero_num_list[node_type_mapping[change] + 1]
                        
                        temp[change] = np.random.randint(
                            int(start), int(end), 1) + 1
                else:
                    
                    if len(num_list) == 1:
                        # Only one node type
                        temp = np.random.randint(
                            0, max_id, sample.shape[-1]) + 1
                    
                    else:
                        for k in range(temp.shape[-1]):
                            start = zero_num_list[node_type_mapping[k]]
                            end = zero_num_list[node_type_mapping[k] + 1]
                            temp[k] = np.random.randint(
                                int(start), int(end), 1) + 1
                
                temp.sort()
                a = set([tuple(temp)])
            
            if len(temp) > 0:
                neg_list.append(temp)
                if i == 0:
                    new_index.append(j)
    if get_type == 'all' or get_type == 'edge':
        x_e, neg_e = generate_negative_edge(x, int(len(x)))
        if get_type == 'all':
            x = list(x) + x_e
            neg_list = neg_list + neg_e
        else:
            x = x_e
            neg_list = neg_e
    new_index = np.array(new_index)
    new_x = x[new_index]
    
    if not forward:
        device = 'cpu'
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    new_weight = torch.tensor(weight[new_index]).to(device)
    
    x = np2tensor_hyper(new_x, dtype=torch.long)
    neg = np2tensor_hyper(neg_list, dtype=torch.long)
    x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
    neg = pad_sequence(neg, batch_first=True, padding_value=0).to(device)
    
    return torch.cat([x, neg]), torch.cat(
        [torch.ones((len(x), 1), device=device), torch.zeros((len(neg), 1), device=device)], dim=0), torch.cat(
        ((torch.ones((len(x), 1), device=device) * new_weight.view(-1, 1), (torch.ones((len(neg), 1), device=device)))))


def save_embeddings(model, origin=False):
    model.eval()
    with torch.no_grad():
        ids = np.arange(num_list[-1].item()) + 1
        ids = torch.Tensor(ids).long().to(device).view(-1, 1)
        embeddings = []
        for j in range(math.ceil(len(ids) / batch_size)):
            x = ids[j * batch_size:min((j + 1) * batch_size, len(ids))]
            if origin:
                embed = model.get_node_embeddings(x)
            else:
                embed = model.get_embedding_static(x)
            embed = embed.detach().cpu().numpy()
            embeddings.append(embed)
        
        embeddings = np.concatenate(embeddings, axis=0)[:, 0, :]
        for i in range(len(num_list)):
            start = 0 if i == 0 else num_list[i - 1]
            static = embeddings[int(start):int(num_list[i])]
            np.save("../mymodel_%d.npy" % (i), static)
            
            if origin:
                np.save("../mymodel_%d_origin.npy" % (i), static)
    
    torch.cuda.empty_cache()
    return embeddings


def generate_H(edge, nums_type, weight):
    nums_examples = len(edge)
    H = [0 for i in range(len(nums_type))]
    if len(nums_type) > 1:
        for i in range(edge.shape[-1]):
        # np.sqrt(weight) because the dot product later would recovers it
            H[node_type_mapping[i]] += csr_matrix((np.sqrt(weight), (edge[:, i], range(
                nums_examples))), shape=(nums_type[node_type_mapping[i]], nums_examples))
    else:
        for i in range(edge.shape[-1]):
            # np.sqrt(weight) because the dot product later would recovers it
            col = [i for x in range(len(edge[i]))]
            data = [np.sqrt(weight)[0] for x in range(len(edge[i]))]
            H[0] += csr_matrix((data, (edge[i], col)),
                               shape=(nums_type[0], nums_examples))
    return H


def generate_H_intact(edge, nums_type, weight):
    row, col = [], []
    if len(nums_type) > 1:
        for i in range(edge.shape[0]):
            row = np.append(row, edge[i]-1)
            col += [i,i,i]
    else:
        for i in range(edge.shape[0]):
            row = np.append(row, edge[i] - 1)
            col += [i for x in range(len(edge[i]))]

    data = [1.0 for i in range(len(row))]
    H = csr_matrix((data, (row, col)), shape=(sum(nums_type).item(), len(edge)))
    return H


def generate_G_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H))
        return G

    
def _generate_G_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :return: G
    """
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = H * W
    # the degree of the hyperedge
    DE = np.squeeze(np.asarray(np.sum(H, axis=0)))
   
    invDE = np.power(DE, -1)
    DV2 = np.power(DV, -0.5)
    invDE[invDE == inf] = 0
    DV2[DV2 == inf] = 0

    HT = H.transpose()

    WinvDE = diags(W*invDE)
    DV2 = csr_matrix(np.mat(np.diag(DV2)))
    G = DV2 * H * WinvDE * HT * DV2

    return G


def generate_seperate_G(G, num_list):
    print(num_list)
    idx_1, idx_2, idx_3 = num_list[0].item(), num_list[1].item(), num_list[2].item()
    G_23 = G[:idx_1, idx_1:]
    index_23 = np.concatenate([np.arange(0, idx_1), np.arange(idx_2, idx_3)])
    G_13 = G[idx_1:idx_2, index_23]
    G_12 = G[idx_2:,:idx_2]
    print(G_23, G_13, G_12)
    return G_23, G_13, G_12


def generate_embeddings(edge, nums_type, H=None, weight=1):
    if H is None:
        H = generate_H(edge, nums_type, weight)

    embeddings = [H[i].dot(s_vstack([H[j] for j in range(len(num))]).T).astype('float32') for i in
                  range(len(num))]
    
    new_embeddings = []
    zero_num_list = [0] + list(num_list)

    for i, e in enumerate(embeddings):
        # This is to remove diag entrance
        for j, k in enumerate(range(zero_num_list[i], zero_num_list[i + 1])):
            e[j, k] = 0
        # Automatically removes all zero entries
        col_sum = np.array(e.sum(0)).reshape((-1))
        new_e = e[:, col_sum > 0]
        new_e.eliminate_zeros()
        new_embeddings.append(new_e)

    # 0-1 scaling
    for i in range(len(nums_type)):
        col_max = np.array(new_embeddings[i].max(0).todense()).flatten()
        _, col_index = new_embeddings[i].nonzero()
        new_embeddings[i].data /= col_max[col_index]

    return [new_embeddings[i] for i in range(len(nums_type))]


def get_adjacency(data, norm=True):
    A = np.zeros((num_list[-1], num_list[-1]))
    
    for datum in tqdm(data):
        for i in range(datum.shape[-1]):
            for j in range(datum.shape[-1]):
                if i != j:
                    A[datum[i], datum[j]] += 1.0
    
    if norm:
        temp = np.concatenate((np.zeros((1), dtype='int'), num), axis=0)
        temp = np.cumsum(temp)
        
        for i in range(len(temp) - 1):
            A[temp[i]:temp[i + 1],
            :] /= (np.max(A[temp[i]:temp[i + 1],
                          :],
                          axis=0,
                          keepdims=True) + 1e-10)
    
    return csr_matrix(A).astype('float32')

args = parse_args()
neg_num = 5
batch_size = 96
neg_num_w2v = 5
bottle_neck = args.dimensions
pair_ratio = 0.9
train_type = 'hyper'

hid_dim = args.dimensions
drop_out = args.dropout

if args.data_split == 'less':
    train_zip = np.load("../data/%s/train_data_25.npz" % (args.data), allow_pickle=True)
    test_zip = np.load("../data/%s/test_data_25.npz" % (args.data), allow_pickle=True)
elif args.data_split == 'more':
    train_zip = np.load("../data/%s/train_data_15.npz" % (args.data), allow_pickle=True)
    test_zip = np.load("../data/%s/test_data_15.npz" % (args.data), allow_pickle=True)
elif args.data_split == 'norm':
    train_zip = np.load("../data/%s/train_data_20.npz" % (args.data), allow_pickle=True)
    test_zip = np.load("../data/%s/test_data_20.npz" % (args.data), allow_pickle=True)
elif args.data_split == 'half':
    train_zip = np.load("../data/%s/train_data_50.npz" % (args.data), allow_pickle=True)
    test_zip = np.load("../data/%s/test_data_50.npz" % (args.data), allow_pickle=True)
else:
    train_zip = np.load("../HNHN_data/%s/train_data.npz" % (args.data), allow_pickle=True)
    test_zip = np.load("../HNHN_data/%s/test_data.npz" % (args.data), allow_pickle=True)
train_data, test_data = train_zip['train_data'], test_zip['test_data']

hypergraph = train_data
try:
    train_weight, test_weight = train_zip["train_weight"].astype('float32'), test_zip["test_weight"].astype('float32')
except BaseException:
    print("no specific train weight")
    test_weight = np.ones(len(test_data), dtype='float32')
    train_weight = np.ones(len(train_data), dtype='float32') * neg_num

new_data = ['dblp', 'cora', 'cora-cite', 'citeseer', 'pubmed']
if args.data in new_data:
    with open("../HNHN_data/%s/features.pickle" % (args.data), 'rb') as file:
        features = pickle.load(file)

num = train_zip['nums_type']
num_list = np.cumsum(num)
print("Node type num", num)


if len(num) > 1:
    node_type_mapping = [0, 1, 2]
else:
    node_type_mapping = [0]

if args.feature == 'adj':
    if args.data in new_data:
        structure_initial = generate_embeddings(train_data, num, H=None, weight=train_weight)
        if args.input == 'original':
            embeddings_initial = structure_initial
        elif args.input == 'feature':
            embeddings_initial = [features]
        elif args.input == 'concat':
            embeddings_initial = [sparse.hstack((features, structure_initial[0]))] #[features]#structure_initial #[sparse.hstack((features, structure_initial[0]))]
    else:
        embeddings_initial = generate_embeddings(train_data, num, H=None, weight=train_weight)


print(train_weight)
print(train_weight, np.min(train_weight), np.max(train_weight))
train_weight_mean = np.mean(train_weight)
train_weight = train_weight / train_weight_mean * neg_num
test_weight = test_weight / train_weight_mean * neg_num


# Now for multiple node types, the first column id starts at 0, the second
# starts at num_list[0]...
if len(num) > 1:
    for i in range(len(node_type_mapping) - 1):
        train_data[:, i + 1] += num_list[node_type_mapping[i + 1] - 1]
        test_data[:, i + 1] += num_list[node_type_mapping[i + 1] - 1]

num = torch.as_tensor(num)
num_list = torch.as_tensor(num_list)

# Add 1 for the padding index
print("adding pad idx")
train_data = add_padding_idx(train_data)
test_data = add_padding_idx(test_data)

# Note that, no matter how many node types are here, make sure the
# hyperedge (N1,N2,N3,...) has id, N1 < N2 < N3...
train_dict = parallel_build_hash(train_data, "build_hash", args, num, initial = set())
test_dict = parallel_build_hash(test_data, "build_hash", args, num, initial = train_dict)
print ("dict_size", len(train_dict), len(test_dict))
print("train data amount", len(train_data))

out_f = 1

with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        if args.feature == 'adj':
            print(embeddings_initial)
            flag = False
            node_embedding = MultipleEmbedding(
                embeddings_initial,
                bottle_neck,
                flag,
                num_list,
                node_type_mapping).to(device)

        elif args.feature == 'attn':
            H = generate_H_intact(train_data, num, train_weight)

            embeddings = (H * H.transpose())
            embeddings.setdiag(neg_num)
            embeddings = embeddings.tocoo()
            embeddings = torch.sparse.FloatTensor(torch.LongTensor([embeddings.row.tolist(), embeddings.col.tolist()]),
                                                  torch.FloatTensor(embeddings.data)).to(device)

            G = generate_G_from_H(H)  
            G = G.tocoo()
            G = torch.sparse.FloatTensor(torch.LongTensor([G.row.tolist(), G.col.tolist()]),
                                         torch.FloatTensor(G.data)).to(device)
            node_embedding = Attn_Embedding(input_dim=embeddings.shape[1],
                                            out_dim=args.dimensions, hidden_dim=hid_dim,
                                            dropout=drop_out, filter_mtx=G, embed=embeddings, num_list=num)

        elif args.feature == 'conv':
    
            H = generate_H_intact(train_data, num, train_weight)

            if args.data in new_data:
                X = features.todense()
                X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
                X = torch.FloatTensor(np.array(X.todense()))

                embeddings = features.tocoo()
                embeddings = torch.sparse.FloatTensor(
                    torch.LongTensor([embeddings.row.tolist(), embeddings.col.tolist()]),
                    torch.FloatTensor(embeddings.data), torch.Size(embeddings.shape)).to(device)

                G = generate_G_from_H(H) 
                G = G.tocoo()
                G = torch.sparse.FloatTensor(torch.LongTensor([G.row.tolist(), G.col.tolist()]),
                                             torch.FloatTensor(G.data), torch.Size(G.shape)).to(device)
            else:
                embeddings = (H * H.transpose())
                embeddings = embeddings.tocoo()
                embeddings = torch.sparse.FloatTensor(torch.LongTensor([embeddings.row.tolist(), embeddings.col.tolist()]),
                                              torch.FloatTensor(embeddings.data),torch.Size(embeddings.shape)).to(device)

                G = generate_G_from_H(H)
                G = G.tocoo()
                G = torch.sparse.FloatTensor(torch.LongTensor([G.row.tolist(), G.col.tolist()]),
                                              torch.FloatTensor(G.data),torch.Size(G.shape)).to(device)
            out_f = args.layers

            if args.layers == 1:
                node_embedding = Conv_Embedding_1(input_dim=embeddings.shape[1],
                                                  out_dim=args.dimensions, hidden_dim=hid_dim,
                                                  dropout=drop_out, filter_mtx=G, embed=embeddings, num_list=num)
            elif args.layers == 2:
                if args.type == 'concat':
                    node_embedding = Conv_Embedding_2(input_dim=embeddings.shape[1],
                                                    out_dim=args.dimensions, hidden_dim=hid_dim,
                                                    dropout=drop_out, filter_mtx=G, embed=embeddings, num_list=num)
                else:
                    node_embedding = Conv_Embedding_2_add(input_dim=embeddings.shape[1],
                                                        out_dim=args.dimensions, hidden_dim=hid_dim,
                                                        dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                        num_list=num, rezero=args.rezero)
                    out_f = 1
            elif args.layers == 3:
                if args.type == 'concat':
                    if not args.dense:
                        print("3-layer GCN not dense")
                        node_embedding = Conv_Embedding_3(input_dim=embeddings.shape[1],
                                                          out_dim=args.dimensions, hidden_dim=hid_dim,
                                                          dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                          num_list=num, rezero=args.rezero)

                    else:
                        print("3-layer GCN Dense")
                        node_embedding = Conv_Embedding_3_dense(input_dim=embeddings.shape[1],
                                                                out_dim=args.dimensions, hidden_dim=hid_dim,
                                                                dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                                num_list=num)
                else:
                    node_embedding = Conv_Embedding_3_add(input_dim=embeddings.shape[1],
                                                          out_dim=args.dimensions, hidden_dim=hid_dim,
                                                          dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                          num_list=num, rezero=args.rezero)
                    out_f = 1
            elif args.layers == 4:
                if args.type == 'concat':
                    print("4-layer GCN not dense")
                    node_embedding = Conv_Embedding_4(input_dim=embeddings.shape[1],
                                                      out_dim=args.dimensions, hidden_dim=hid_dim,
                                                      dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                      num_list=num, rezero=args.rezero)

        if args.data in new_data and args.feature == 'conv':
            classifier_model = Classifier_Attn(
                n_head=8,
                d_model=args.dimensions,
                d_k=16,
                d_v=16,
                node_embedding=node_embedding,
                diag_mask=args.diag,
                bottle_neck=bottle_neck,
                out_f=out_f).to(device)
        else:
            classifier_model = Classifier_Attn(
                n_head=8,
                d_model=args.dimensions,
                d_k=16,
                d_v=16,
                node_embedding=node_embedding,
                diag_mask=args.diag,
                bottle_neck=bottle_neck,
                out_f=out_f).to(device)

        loss = F.binary_cross_entropy

        params_list = list(classifier_model.parameters())
        
        if args.feature == 'adj' or args.feature == 'conv' or args.feature == 'sepconv':
            optimizer = torch.optim.Adam(params_list, lr=1e-3, weight_decay=5e-4)
        else:
            optimizer = torch.optim.RMSprop(params_list, lr=1e-3)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100],
                                                         gamma=0.9)

        model_parameters = filter(lambda p: p.requires_grad, params_list)
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("params to be trained", params)

        if args.test:
            test(args, classifier_model,
                 loss=loss,
                 validation_data=(test_data, test_weight),
                batch_size=batch_size)
        else:
            train(args, classifier_model,
                  loss=loss,
                  training_data=(train_data, train_weight),
                  validation_data=(test_data, test_weight),
                  optimizer=[optimizer], epochs=args.num_epoch, batch_size=batch_size, scheduler=scheduler)
            test(args, classifier_model,
                 loss=loss,
                 validation_data=(test_data, test_weight),
                 batch_size=batch_size)
