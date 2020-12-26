import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from SEAL import util_functions
from pytorch_DGCNN import main
from main import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('PDF')


parser = argparse.ArgumentParser(description='Link Prediction for AOPEDF')
# general settings
parser.add_argument('--data-name', default=None, help='network name')
#parser.add_argument('--train-name', default=None, help='train name')
#parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predictions\
                    for links in test-name; you still need to specify train-name\
                    in order to build the observed network and extract subgraphs')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.2,
                    help='ratio of test links')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                    help='if True, regard all unknown links as negative test data; \
                    sample a portion from them as negative training data. Otherwise,\
                    train negative and test negative data are both sampled from \
                    unknown links without overlap.')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use metapath2vec node embeddings')
parser.add_argument('--which-embedding', default='node2vec', help='node2vec & meta & node & meta_node')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='save the final model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
data = sio.loadmat(args.data_dir)
net = data['net']
if 'group' in data:
    # load node attributes (here a.k.a. node classes)
    attributes = data['group'].toarray().astype('float32')
else:
    attributes = None

# sample both positive and negative train links from net
train_pos, train_neg, test_pos , test_neg = util_functions.sample_neg(
    net, args.test_ratio, max_train_num=args.max_train_num
)
print('already sampled net')
#print(train_pos)
#print(train_neg)

'''Train SEAL model'''
A = net.copy()  # the observed network
A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x4
node_information = None
if args.use_embedding:
    embeddings = util_functions.generate_node2vec_embeddings(A, args.embed_dim, True, train_neg, args.which_embedding)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

train_graphs, test_graphs, max_n_label = util_functions.links2subgraphs(
    A, 
    train_pos, 
    train_neg, 
    test_pos, 
    test_neg, 
    args.hop, 
    args.max_nodes_per_hop, 
    node_information, 
    False
)
print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

# DGCNN configurations
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.num_epochs = 50
cmd_args.learning_rate = 1e-4
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
    cmd_args.sortpooling_k = max(10, num_nodes_list[k_])
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

classifier = main.Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

random.shuffle(train_graphs)
val_num = int(0.1 * len(train_graphs))
val_graphs = train_graphs[:val_num]
train_graphs = train_graphs[val_num:]

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss,_,_ = main.loop_dataset(
        train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=args.batch_size
    )
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f aupr %.5f\033[0m' % (
        epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3]))

    classifier.eval()
    val_loss,_,_ = main.loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
    print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f aupr %.5f\033[0m' % (
        epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3]))
    if best_loss is None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        best_epoch = epoch
        test_loss,precision, recall = main.loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f aupr %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))
        plt.figure()
        plt.plot(recall, precision , 'b', label='AUPR = %0.3f %%' % test_loss[3]*100)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.35, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve')
        plt.legend(loc="lower right")
        plt.savefig('./figure/fig%d.pdf' %epoch)

print('\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f aupr %0.5f\033[0m' % (
    best_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]))

if args.save_model:
    model_name = 'data/{}_model.pth'.format(args.data_name)
    print('Saving final model states to {}...'.format(model_name))
    torch.save(classifier.state_dict(), model_name)
    hyper_name = 'data/{}_hyper.pkl'.format(args.data_name)
    with open(hyper_name, 'wb') as hyperparameters_file:
        pickle.dump(cmd_args, hyperparameters_file)
        print('Saving hyperparameters to {}...'.format(hyper_name))

with open('acc_results.txt', 'a+') as f:
    f.write(str(args.embed_dim)+'_'+str(args.which_embedding) +':  '+str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(args.embed_dim)+'_'+str(args.which_embedding) +':  '+str(test_loss[2]) + '\n')
    with open('aupr_results.txt', 'a+') as f:
    	f.write(str(args.embed_dim)+'_'+str(args.which_embedding) +':  '+str(test_loss[3]) + '\n')

