import os
import numpy as np
from PIL import Image
from collections import defaultdict
from scipy.spatial.distance import cdist
from nltk.tokenize import word_tokenize

def load_DA_SB_cost(name, n=None, norm_cost=1, metric = 'sqeuclidean', scale_factor=1, seed=0, nlp_i=None, nlp_name=None, nlp_portion_size=None):
    switcher = {
        'synthetic_OT': get_synthetic_input_OT,
        'synthetic_matching': get_synthetic_input_matching,
        'mnist_OT': get_mnist_input_OT,
        'mnist_matching': get_mnist_input_matching,
        'NLP_OT': get_NLP_input_OT_ori,
        'NLP_OT_precal': get_NLP_input_OT_precal,
    }
    func = switcher.get(name)
    if name == 'synthetic_OT':
        return func(n, norm_cost, metric, seed)
    elif name == 'synthetic_matching':
        return func(n, norm_cost, metric, seed)
    elif name == 'mnist_OT':
        return func(metric, scale_factor, norm_cost, seed)
    elif name == 'mnist_matching':
        return func(n, norm_cost, metric, seed)
    elif name =='NLP_OT_precal':
        return func(nlp_i, norm_cost)
    elif name == 'NLP_OT':
        return func(nlp_name, norm_cost, metric, nlp_portion_size, seed)
    else:
        print("Invalid Data Name")
        return None

def get_synthetic_input_OT(n = 1000, norm_cost=1, metric = 'sqeuclidean', seed = 0):
    """
    This function creates synthetic experiment data by randomly generating points in a 2d unit square.
    """
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, metric)
    C = cost.max()
    if norm_cost:
        cost = cost/C
    DA = np.random.rand(n)
    SB = np.random.rand(n)
    DA = DA/np.sum(DA)
    SB = SB/np.sum(SB)
    return DA, SB, cost

def get_mnist_input_OT(metric='sqeuclidean', scale_factor=1, norm_cost = 1, seed = 0):
    np.random.seed(seed)
    if os.path.exists('mnist.npy'):
        mnist = np.load('mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8

    ran_int = np.random.randint(10, high=None, size=2)
    a_indx = np.where(mnist_labels == ran_int[0])
    b_indx = np.where(mnist_labels == ran_int[1])
    a = mnist[np.random.choice(a_indx[0], size=1),:,:]
    b = mnist[np.random.choice(b_indx[0], size=1),:,:]

    m = a.shape[-1]
    if scale_factor != 1:
        im_a = Image.fromarray(np.uint8(a[0,:]))
        m *= scale_factor
        im_a = im_a.resize((m, m))
        a = np.asarray(im_a)
        im_b = Image.fromarray(np.uint8(b[0,:]))
        im_b = im_b.resize((m, m))
        b = np.asarray(im_b)

    print("problem size = {}".format(m*m))
    a = a/255
    b = b/255
    a = a.reshape(-1, m*m)
    b = b.reshape(-1, m*m)

    x = np.repeat(np.arange(m),m)
    y = np.tile([np.arange(m)],m)
    coord = np.transpose(np.vstack((x,y)))

    coord_a = coord[a[0,:]!=0,:]
    coord_b = coord[b[0,:]!=0,:]
    cost = cdist(coord_b, coord_a, metric)

    C = cost.max()
    if norm_cost:
        cost = cost/C

    a = a[a!=0]
    b = b[b!=0]
    a = np.squeeze(a/np.sum(a))
    b = np.squeeze(b/np.sum(b))

    return a, b, cost

def get_synthetic_input_matching(n = 1000, norm_cost=1, metric = 'sqeuclidean', seed = 0):
    """
    This function creates synthetic experiment data by randomly generating points in a 2d unit square.
    """
    np.random.seed(seed)
    a = np.random.rand(n,2)
    b = np.random.rand(n,2)
    cost = cdist(a, b, metric)
    C = cost.max()
    if norm_cost:
        cost = cost/C
    DA = np.ones(n)/n
    SB = np.ones(n)/n
    return DA, SB, cost

def get_mnist_input_matching(n=1000, norm_cost=1, metric = 'minkowski', seed = 0):

    if os.path.exists('./mnist.npy'):
        mnist = np.load('./mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('./mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8

    np.random.seed(seed)

    total = np.arange(len(mnist_labels))
    indx_a = total[mnist_labels < 5]
    indx_b = total[mnist_labels > 4]

    indx_a = np.random.permutation(indx_a)[:n]
    indx_b = np.random.permutation(indx_b)[:n]

    a  = mnist[indx_a, :, :]
    b  = mnist[indx_b, :, :]

    # im2double
    a = a/255.0
    b = b/255.0
    a = a.reshape(-1, 784)
    b = b.reshape(-1, 784)
    a = a / a.sum(axis=1, keepdims=1)
    b = b / b.sum(axis=1, keepdims=1)

    cost = cdist(a, b, metric='minkowski', p=1)
    C = cost.max()
    if norm_cost:
        cost = cost/C
    
    DA = np.ones(n)/n
    SB = np.ones(n)/n

    return DA, SB, cost

def get_NLP_input_OT_precal(nlp_i = 1, norm_cost=1):
    with open('./data/NLP/NLP{}_new.txt'.format(nlp_i), 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                shape = line.split(' ')
                shape = np.array(shape, dtype=int)
                print("problem size={}".format(shape[0]*shape[1]))
            elif i == 1:
                SB = line.split(' ')
                SB = np.array(SB, dtype=float)
                print(len(SB))
            elif i == 2:
                DA = line.split(' ')
                DA = np.array(DA, dtype=float)
                print(len(DA))
            else:
                cost = line.split(' ')
                cost = np.array(cost, dtype=float)
                print(len(cost))
                cost = cost.reshape(shape[0], shape[1])

    DA = DA/np.sum(DA)
    SB = SB/np.sum(SB)

    C = cost.max()
    if norm_cost:
        cost = cost/C
    
    return DA, SB, cost

def count_tokens(tokenized_text):
    # Count the number of appearances for all unique tokens
    counts = defaultdict(int)
    for token in tokenized_text:
        counts[token] += 1
    return counts

def get_NLP_input_OT_ori(text_name, norm_cost=1, metric = 'euclidean', portion_size = 100, seed = 0):
    np.random.seed(seed)
    # Load the novel text
    with open("{}.txt".format(text_name), "r") as f:
        text = f.read()

    # Split text into disjoint portions
    lines = text.split("\n")
    text_portions = [lines[i:i+portion_size] for i in range(0, len(lines), portion_size)]
    text_portions = ["\n".join(portion) for portion in text_portions]

    text_num = len(text_portions)
    rand_ind = np.random.randint(text_num, high=None, size=2, dtype=int)
    text_portions = [text_portions[rand_ind[0]], text_portions[rand_ind[1]]]

    # Tokenize text using NLTK
    tokenized_portions = [word_tokenize(portion) for portion in text_portions]

    # Output unique tokens and corresponding number of appearances
    DA = count_tokens(tokenized_portions[0])
    SB = count_tokens(tokenized_portions[1])
    tokenized_portions = [list(DA.keys()), list(SB.keys())]
    DA = np.array(list(DA.values()), dtype=int)
    SB = np.array(list(SB.values()), dtype=int)
    DA = DA/np.sum(DA)
    SB = SB/np.sum(SB)

    # Load GloVe word embeddings
    glove = {}
    with open('glove.6B.100d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            glove[word] = vectors

    # Create word embeddings for each distribution
    distribution_embeddings = []
    for portion in tokenized_portions:
        portion_vectors = np.zeros((len(portion), 100))
        for i, word in enumerate(portion):
            if word in glove:
                portion_vectors[i] = glove[word]
        distribution_embeddings.append(portion_vectors)

    cost = cdist(distribution_embeddings[1], distribution_embeddings[0], metric)
    if norm_cost:
        C = cost.max()
        cost = cost/C
    return DA, SB, cost