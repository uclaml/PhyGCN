from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary
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
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--data', type=str, default='dblp')

    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')

    parser.add_argument('-i', '--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout hyperparameter. Default is 0.5')

    parser.add_argument('-d', '--diag', type=str, default='True',
                        help='Use the diag mask or not')
    parser.add_argument(
        '-f',
        '--feature',
        type=str,
        default='conv',
        help='Features used')
    parser.add_argument('--num-epoch', type=int, default=300,
                        help='Number of epochs. Default is 300.')

    parser.add_argument('--layers', default=2, type=int,
                        help='Layers of hyperGCN')
    parser.add_argument('--type', default='concat', type=str,
                        help='How to combine the outputs')
    parser.add_argument('--rezero', default=0, type=int,
                        help='Using rezero')
    parser.add_argument('--dropedge', default=1.0, type=float,
                        help='Percentage of edges to keep for each epoch')
    parser.add_argument('--split', default=1, type=int,
                        help='Different data split')

    args = parser.parse_args()

    args.epoch = 25

    args.save_path = os.path.join(
        'checkpoints/', args.data)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def train_batch_hyperedge(model, loss_func, batch_data, y):
    x = batch_data
    pred, recon_loss = model(x, return_recon=True)
    loss = loss_func(pred, y)
    return pred, y, loss


def accuracy_one_hot(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels
    returns:
    accuracy
    """
    predictions = Z.max(1)[1].type_as(Y)
    correct = predictions.eq(Y).double()
    correct = correct.sum()

    accuracy = correct / len(Y)
    return accuracy.item()


def train_epoch(args, model, loss_func, training_data, optimizer, batch_size):
    data, y = training_data

    # Permutate all the data
    index = torch.randperm(len(data))
    data, y = data[index], y[index]

    # dropedge
    if args.dropedge < 1.0:
        G = model.node_embedding.G  
        rowcol = model.node_embedding.G._indices()

        val = model.node_embedding.G._values()
        idx = random.sample(range(len(val)), int(args.dropedge * len(val)))
        model.node_embedding.G = torch.sparse.FloatTensor(rowcol[:, idx], val[idx],
                                                          torch.Size(model.node_embedding.G.shape)).to(device)

    model.train()
    bce_total_loss = 0
    acc_list, y_list, pred_list = [], [], []

    batch_num = int(math.floor(len(data) / batch_size))
    bar = trange(batch_num, mininterval=0.1, desc='  - (Training) ', leave=False, )
    for i in bar:

        batch_data = data[i * batch_size:(i + 1) * batch_size]
        batch_y = y[i * batch_size:(i + 1) * batch_size]

        pred, batch_y, loss_bce = train_batch_hyperedge(model, loss_func, batch_data, y=batch_y)
        acc_list.append(accuracy_one_hot(pred, batch_y))
        y_list.append(batch_y)
        pred_list.append(pred)

        for opt in optimizer:
            opt.zero_grad()

        # backward
        loss_bce.backward()
        # update parameters
        for opt in optimizer:
            opt.step()

        bce_total_loss += loss_bce.item()

    pred = torch.cat(pred_list)

    if args.dropedge < 1.0:
        model.node_embedding.G = G

    return bce_total_loss / batch_num, np.mean(acc_list)


def eval_epoch(args, model, loss_func, validation_data, verbose=False):
    ''' Epoch operation in evaluation phase '''
    bce_total_loss = 0

    model.eval()
    with torch.no_grad():
        validation_data, y = validation_data

        index = torch.randperm(len(validation_data))
        validation_data, y = validation_data[index], y[index]
        batch_size = len(validation_data)
        pred, label = [], []

        for i in tqdm(range(int(math.floor(len(validation_data) / batch_size))),
                      mininterval=0.1, desc='  - (Validation)   ', leave=False):
            # prepare data
            batch_x = validation_data[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]

            index = torch.randperm(len(batch_x))
            batch_x, batch_y = batch_x[index], batch_y[index]

            pred_batch, _ = model(batch_x, return_recon=True)
            pred.append(pred_batch)
            label.append(batch_y)
            loss = loss_func(pred_batch, batch_y)

            bce_total_loss += loss.item()

        pred = torch.cat(pred, dim=0)
        label = torch.cat(label, dim=0)
        acc = accuracy_one_hot(pred, label)

        if verbose:
            predictions = pred.max(1)[1].type_as(label)
            correct = predictions.eq(label)
            torch.save(validation_data[correct], args.data+'correct.pt')

    return bce_total_loss / (i + 1), acc


def train(args, model, loss, training_data, validation_data, optimizer, epochs, batch_size,
          scheduler):  # scheduler
    valid_accus = [0]

    for epoch_i in range(epochs):

        bce_loss, train_accu = train_epoch(
            args, model, loss, training_data, optimizer, batch_size)

        valid_bce_loss, valid_accu = eval_epoch(args, model, loss, validation_data)
        scheduler.step()

        checkpoint = {
            'model_link': model.state_dict(),
            'epoch': epoch_i}

        model_name = 'model_conv_node.chkpt'

        if valid_accu >= max(valid_accus):
            torch.save(checkpoint, os.path.join(args.save_path, model_name))

        valid_accus += [valid_accu]
        torch.cuda.empty_cache()

    checkpoint = torch.load(os.path.join(args.save_path, model_name))
    model.load_state_dict(checkpoint['model_link'])
    #print("Saved at epoch # ", checkpoint['epoch'], file=open("result.txt", "a"))
    print("Max Accuracy: ", max(valid_accus))

    valid_bce_loss, valid_accu = eval_epoch(args, model, loss, validation_data, verbose=True)
    print(str(valid_accu)+',', file=open("result.txt", "a"))


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


def generate_H_intact(edge, nums_type):
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
    H = csr_matrix((data, (row, col)), shape=(sum(nums_type), len(edge)))
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


args = parse_args()

bottle_neck = args.dimensions
hid_dim = args.dimensions
drop_out = args.dropout

new_data = ['dblp', 'cora', 'cora-cite', 'citeseer', 'pubmed']
if args.data in new_data:
    with open("HNHN_data/%s/features.pickle" % (args.data), 'rb') as file:
        features = pickle.load(file)
    with open("HNHN_data/%s/labels.pickle" % (args.data), 'rb') as file:
        labels = pickle.load(file)
    with open("HNHN_data/%s/splits/splits_02/%s.pickle" % (args.data, str(args.split)), 'rb') as file:
        split = pickle.load(file)
    with open("HNHN_data/%s/hypergraph.pickle" % (args.data), 'rb') as file:
        hypergraph = pickle.load(file)

    labels = np.array(labels)
    batch_size = len(split['train'])

    node_train_data, node_test_data = split['train'], split['test']
    node_train_label, node_test_label = labels[split['train']], labels[split['test']]

    node_train_data, node_test_data = np.array(node_train_data)+1, np.array(node_test_data)+1
    node_train_label, node_test_label = np.array(node_train_label), np.array(node_test_label)

    node_train_data, node_test_data = torch.tensor(node_train_data).to(device), torch.tensor(node_test_data).to(device)
    node_train_label, node_test_label = torch.tensor(node_train_label).to(device), torch.tensor(node_test_label).to(device)

    print(len(node_train_data),len(node_test_data))

    num = [len(labels)]

    edges = np.array([list(x) for x in hypergraph.values()])
    edges = add_padding_idx(edges)

if args.data == 'dblp' or args.data == 'citeseer':
    classes = 6
elif args.data == 'pubmed':
    classes = 3
elif args.data == 'cora' or args.data == 'cora-cite':
    classes = 7

node_type_mapping = [0, 1, 2]

with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):


        X = features.todense()
        X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
        X = torch.FloatTensor(np.array(X.todense()))

        features = features.tocoo()
        embeddings = torch.sparse.FloatTensor(torch.LongTensor([features.row.tolist(), features.col.tolist()]),
                                                  torch.FloatTensor(features.data), torch.Size((num[0], features.shape[-1]))).to(device)

        H = generate_H_intact(edges, num)
        torch.save(H,'H.pt')
        G = generate_G_from_H(H)
        G = G.tocoo()
        G = torch.sparse.FloatTensor(torch.LongTensor([G.row.tolist(), G.col.tolist()]),
                                         torch.FloatTensor(G.data), torch.Size(G.shape)).to(device)
        out_f = args.layers
        num = torch.tensor(num).to(device)

        if args.layers == 1:
            node_embedding = Conv_Embedding_1(input_dim=embeddings.shape[1],
                                                  out_dim=args.dimensions, hidden_dim=hid_dim,
                                                  dropout=drop_out, filter_mtx=G, embed=embeddings, num_list=num)
        elif args.layers == 2:
            if args.type == 'concat':
                node_embedding = Conv_Embedding_2(input_dim=embeddings.shape[1],
                                                      out_dim=args.dimensions, hidden_dim=hid_dim,
                                                      dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                      num_list=num)
            else:
                node_embedding = Conv_Embedding_2_add(input_dim=embeddings.shape[1],
                                                          out_dim=args.dimensions, hidden_dim=hid_dim,
                                                          dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                          num_list=num, rezero=args.rezero)
                out_f = 1
        elif args.layers == 3:
            if args.type == 'concat':
                print("3-layer GCN with concatenation")
                node_embedding = Conv_Embedding_3(input_dim=embeddings.shape[1],
                                                      out_dim=args.dimensions, hidden_dim=hid_dim,
                                                      dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                      num_list=num, rezero=args.rezero)

            else:
                node_embedding = Conv_Embedding_3_add(input_dim=embeddings.shape[1],
                                                          out_dim=args.dimensions, hidden_dim=hid_dim,
                                                          dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                          num_list=num, rezero=args.rezero)
                out_f = 1
        elif args.layers == 4:
            if args.type == 'concat':
                    print("4-layer GCN with concatenation")
                    node_embedding = Conv_Embedding_4(input_dim=embeddings.shape[1],
                                                      out_dim=args.dimensions, hidden_dim=hid_dim,
                                                      dropout=drop_out, filter_mtx=G, embed=embeddings,
                                                      num_list=num, rezero=args.rezero)

        if args.data in new_data and args.feature == 'conv':
            classifier_model = Classifier_Node(
                d_model=args.dimensions,
                node_embedding=node_embedding,
                out_f=out_f,
                c=classes).to(device)
            model_name = 'model_conv_ssl_HNHN.chkpt'
            checkpoint = torch.load(os.path.join(args.save_path, model_name))
            classifier_model.node_embedding.load_state_dict(checkpoint['model_link'])

        loss = F.nll_loss 
        params_list = list(classifier_model.parameters())

        if args.feature == 'adj' or args.feature == 'conv' or args.feature == 'sepconv':
            optimizer = torch.optim.SGD(params_list, lr=0.003, momentum=0.9) 
        else:
            optimizer = torch.optim.RMSprop(params_list, lr=1e-3)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100],
                                                         gamma=0.9)

        model_parameters = filter(lambda p: p.requires_grad, params_list)
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("params to be trained", params)

        train(args, classifier_model,
                loss=loss,
                training_data=(node_train_data, node_train_label),
                validation_data=(node_test_data, node_test_label),
                optimizer=[optimizer], epochs=args.num_epoch, batch_size=batch_size,
                scheduler=scheduler)