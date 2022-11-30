######################################################
# Yasmin Heimann, hyasmin, 311546915
# @description Generative Latent Optimization (GLO) for image generation on the MNIST data set
#              The module trains a generator using a linear and convolutional neural network,
#              that aims to learns class and content latent codes to reconstruct images out of them.
#              The module is trained on the MNIST data set.
######################################################

## IMPORT of packages ##
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import GeneratorForMnistGLO
from datasets import MNIST
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Hyper-parameters #
BATCH_SIZE = 32
EPOCS = 50
WEIGHT_DECAY = 0.001
STD_DEV = 0.3

# generate a run name without a point '.'
if 0.00001 < STD_DEV < 1:
    str_std = '0' + str(STD_DEV).split('.')[1]
else:
    str_std = str(STD_DEV)
RUN_NUM = '_e' + str(EPOCS) + '_b' + str(BATCH_SIZE) + '_std' + str_std + '_diff'

# data parameters
SAMPLES_SIZE = 2000
CLASS_SIZE = 10
LATENT_CODE_DIM = 25
INPUT_DIM = LATENT_CODE_DIM * 2

LABEL = 1
TENSOR = 0


def write_data(data):
    """
    Writes the loss data into a text file as a list
    :param data: the list of training loss
    """
    with open('run' + RUN_NUM + '.txt', 'w') as f:
        f.write('[')
        for d in data:
            f.write(str(d) + ', ')
        f.write(']')
    print(data)


def plot_loss(data):
    """
    Plots the average loss of each epoc in the training process, given in the data
    """
    write_data(data)
    x_axis = list(range(1, EPOCS + 1))
    plt.plot(x_axis, data)
    # set x axis and y axis labels
    plt.xlabel('epocs')
    plt.ylabel('training loss')
    # Set a title of the current axes
    title = 'Training Loss Over ' + str(EPOCS) + ' Epocs, ' + str(BATCH_SIZE) + ' Batch Size, std=' + str(STD_DEV)
    plt.title(title)
    # Display and save the figure
    fig_name = 'loss_run' + RUN_NUM + ".png"
    plt.savefig(fig_name, dpi=600)
    plt.show()


def reconstruct_image(class_opt, content_opt, generator, images, i, writer):
    """
    Gets the generators prediction on the image ans writes it to tensorboard
    :param class_opt: the class embedding with optimization
    :param content_opt: the content embedding with optimization
    :param generator: the model
    :param images: the real images to document in the write
    :param i: the loop number
    :param writer: the writer
    :return: the predicted reconstructed image
    """
    # concatenate the class and content embedding
    input = torch.cat((class_opt, content_opt), dim=1)
    pred = generator(input)
    # log the images using tensorboard
    writer.add_images('training/true_images', images, i)
    writer.add_images('training/pred_images', pred, i)
    return pred


def optimization_step(optimizer, criterions, pred, images):
    """
    Optimizes the batch run in the network using the given optimizer
    :param optimizer: the optimizer to work with
    :param criterions: the losses to use
    :param pred: the predicted images
    :param images: the real images
    :return: the loss value
    """
    optimizer.zero_grad()
    loss = criterions[0](pred, images)
    for c in criterions[1:]:
        loss += c(pred, images)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_glo(train_set, class_embedding, content_embedding, log_dir):
    """
    The main function that trains the generative model.
    :param train_set: a data loader with the train set of the real images
    :param class_embedding: class embedding
    :param content_embedding: content embedding
    :param log_dir: the path to log the results into
    :return:
    """
    # create the model and the relevant objects
    generator = GeneratorForMnistGLO(code_dim=INPUT_DIM)
    generator.train()
    # add content and class optimization
    content_opt = Variable(torch.zeros((BATCH_SIZE, LATENT_CODE_DIM)), requires_grad=True)
    class_opt = Variable(torch.zeros((BATCH_SIZE, LATENT_CODE_DIM)), requires_grad=True)
    optimizer = optim.Adam([{'params': generator.parameters()}, {'params': class_opt},
                            {'params': content_opt, 'weight_decay': WEIGHT_DECAY}])
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)
    epocs_loss = []
    # train the model
    for epoc in range(EPOCS):
        epoc_loss = 0
        for i, data in enumerate(train_set, 0):
            # extract the batch data
            images, labels, indices = data
            class_opt.data = class_embedding[labels]
            content_opt.data = content_embedding[indices]
            noise = torch.normal(mean=0, std=STD_DEV, size=(content_opt.shape))
            content_opt.data += noise

            # reconstruct the image
            prediction = reconstruct_image(class_opt, content_opt, generator, images, i, writer)

            # optimize the generator and latent codes
            epoc_loss += optimization_step(optimizer, [criterion1, criterion2], prediction, images)
        # calculate the average loss for this epoc
        epoc_cur_loss = epoc_loss / len(train_set)
        print("epoc ", (epoc + 1), " loss ", epoc_cur_loss)
        epocs_loss.append(epoc_cur_loss)
        writer.add_scalar('train/loss', epoc_cur_loss, epoc)
    writer.close()
    plot_loss(epocs_loss)


def pre_process_data():
    """
    The function creates an embedding of content and latent codes to match the MNIST data set, taking the first
    SAMPLES_SIZE images.
    :return: a train loader object and the embeddings
    """
    # create the data
    im_data = MNIST(SAMPLES_SIZE)
    train_set = DataLoader(im_data, BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    # create embedded content (2000x25) and class (25x10)
    content_emb = nn.Embedding(num_embeddings=SAMPLES_SIZE, embedding_dim=LATENT_CODE_DIM)
    class_emb = nn.Embedding(num_embeddings=CLASS_SIZE, embedding_dim=LATENT_CODE_DIM)

    content_embedding = content_emb(torch.LongTensor(range(SAMPLES_SIZE)))
    class_embedding = class_emb(torch.LongTensor(range(CLASS_SIZE)))
    return train_set, content_embedding, class_embedding


def run_glo():
    """
    The main function of GLO that trains the generator on the MNIST data set and extract the loss
    curve and the reconstructed images (as a batch)
    Hyper parameters can be changes through the magic numbers above
    :return:
    """
    train_set, content_embedding, class_embedding = pre_process_data()
    run = 'run' + RUN_NUM
    print(run + " in process")
    train_glo(train_set, class_embedding, content_embedding, "./log/" + run)


run_glo()
