import tensorflow as tf
import numpy as np
import pylab as Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import tsne

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls_re', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log_re/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump_re', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
parser.add_argument('--inputfile', default='None', help='Please specify which point cloud to evaluate')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path # to change
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 


# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes=1):
    is_training = False

    with tf.device('/gpu: ' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(1, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        #get model
        net, end_points = MODEL.get_model_add_global(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(net['pc_fc3'], labels_pl, end_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    
    #Creat a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")   

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'net': net,
           'loss': loss}

    eval_one_pointcloud(sess, ops)
    #eval_all_pointcloud(sess, ops)

def eval_all_pointcloud(sess, ops, num_votes=1, topk=1):
    '''
    Code to change....
    '''
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    global_features = []
    labels = np.array([])

    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)
        labels = np.append(labels, current_label)
        print(labels)
        print(current_data.shape)

        file_size = current_data.shape[0]
        print(file_size)

        for pc_idx in range(file_size):
            #print(id_count)
            for vote_idx in range(num_votes):
                rotated_data = provider.rotate_point_cloud_by_angle(
                                        current_data[pc_idx:pc_idx + 1, :, :],
                                        vote_idx/float(num_votes) * np.pi * 2)

                feed_dict = {ops['pointclouds_pl']: rotated_data,
                                ops['labels_pl']: current_label[pc_idx:pc_idx + 1],
                                ops['is_training_pl']: is_training}
                
                loss_val, net_val = sess.run([ops['loss'], ops['net']],
                                                feed_dict = feed_dict)

                global_features.append(np.squeeze(net_val['pc_maxpool']))

    global_features = np.array(global_features)
    print "global_features :: ", global_features.shape
    print "labels :: ", labels.shape

    global_features = tsne.tsne(global_features, 2, global_features.shape[1])
    Plot.scatter(global_features[:,0], global_features[:,1], 30, c = 4 * labels, cmap='jet')
    
    for i, txt in enumerate(labels):
        if i % 10 == 0:
            Plot.annotate(txt, (global_features[i,0], global_features[i,1]))
    
    Plot.show()
    #add_sub3d(fig, global_features, 1, 1, 1, c = np.array(labels))
    #plt.show()



def eval_one_pointcloud(sess, ops, num_votes=1, topk=1):
    id_count = 0
    id_pred = 1200
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)
        print(current_data.shape)

        file_size = current_data.shape[0]
        print(file_size)

        for pc_idx in range(file_size):
            if id_count == id_pred:
                #print(id_count)
                for vote_idx in range(num_votes):
                    rotated_data = provider.rotate_point_cloud_by_angle(
                                            current_data[pc_idx:pc_idx + 1, :, :],
                                            vote_idx/float(num_votes) * np.pi * 2)

                    feed_dict = {ops['pointclouds_pl']: rotated_data,
                                 ops['labels_pl']: current_label[pc_idx:pc_idx + 1],
                                 ops['is_training_pl']: is_training}
                    
                    loss_val, net_val = sess.run([ops['loss'], ops['net']],
                                                 feed_dict = feed_dict)

                    show_output(net_val)
                    '''
                    fig = plt.figure()
                    pc_input = np.squeeze(net_val['pc_input'])
                    add_sub3d(fig, pc_input, 1, 2, 1)

                    pc_conv5 = np.squeeze(net_val['pc_conv5'])
                    max_pos_pc = np.argmax(pc_conv5, 0)
                    print max_pos_pc.shape
                    max_pc = pc_input[max_pos_pc[:], :].copy()
                    print max_pc.shape

                    #add_sub3d(fig, max_pc, 1, 2, 2, c = range(len(max_pos_pc)))
                    add_sub3d(fig, max_pc, 1, 2, 2)
                    plt.show()
                    '''

                    print(loss_val)
                    pred_val = np.argmax(net_val['pc_fc3'], 1)
                    print "pred_val :: ", pred_val
                    print "current_label :: ", current_label[pc_idx:pc_idx + 1]
                    
                return
                
            id_count += 1
            
def show_output(net):
    fig = plt.figure()
    pc_input = np.squeeze(net['pc_input'])
    add_sub3d(fig, pc_input, 1, 2, 1)

    pc_transformed_3 = np.squeeze(net['pc_transformed_3'])
    add_sub3d(fig, pc_transformed_3, 1, 2, 2)

    '''
    fig1 = plt.figure()
    pc_conv1 = np.squeeze(net['pc_conv1'])
    pc_conv1 = tsne.tsne(pc_conv1, 3, pc_conv1.shape[1])
    add_sub3d(fig1, pc_conv1, 1, 2, 1)

    pc_conv2 = np.squeeze(net['pc_conv2'])
    pc_conv2 = tsne.tsne(pc_conv2, 3, pc_conv2.shape[1])
    add_sub3d(fig1, pc_conv2, 1, 2, 2)

    fig2 = plt.figure()
    pc_transformed_64 = np.squeeze(net['pc_transformed_64'])
    pc_transformed_64 = tsne.tsne(pc_transformed_64, 3, pc_transformed_64.shape[1])
    add_sub3d(fig2, pc_transformed_64, 1, 2, 1)

    pc_conv3 = np.squeeze(net['pc_conv3'])
    pc_conv3 = tsne.tsne(pc_conv3, 3, pc_conv3.shape[1])
    add_sub3d(fig2, pc_conv3, 1, 2, 2)
    '''
    '''
    
    pc_input = np.squeeze(net['pc_input'])
    add_sub3d(fig, pc_input, 2, 3, 4)
    pc_input = np.squeeze(net['pc_input'])
    add_sub3d(fig, pc_input, 2, 3, 5)
    pc_input = np.squeeze(net['pc_input'])
    add_sub3d(fig, pc_input, 2, 3, 6)
    '''
    plt.show()


def add_sub3d(fig, data, h = 1, w = 1, idx = 1, c = None,str=None):
    assert(data.shape[1]==3)
    ax = fig.add_subplot(h, w, idx, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = c, cmap='jet')


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)    
    LOG_FOUT.close()
