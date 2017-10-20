import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    net = {}
    net['pc_input'] = point_cloud

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    end_points['transform_3'] = transform
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    input_image = tf.expand_dims(point_cloud_transformed, -1)
    net['pc_transformed_3'] = input_image

    net['pc_conv1'] = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net['pc_conv2'] = tf_util.conv2d(net['pc_conv1'], 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net['pc_conv2'], is_training, bn_decay, K=64)
    end_points['transform_64'] = transform
    net_transformed = tf.matmul(tf.squeeze(net['pc_conv2'], axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])
    net['pc_transformed_64'] = net_transformed

    net['pc_conv3'] = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net['pc_conv4'] = tf_util.conv2d(net['pc_conv3'], 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net['pc_conv5'] = tf_util.conv2d(net['pc_conv4'], 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net['pc_maxpool'] = tf_util.max_pool2d(net['pc_conv5'], [num_point,1],
                             padding='VALID', scope='maxpool')

    net['pc_maxpool'] = tf.reshape(net['pc_maxpool'], [batch_size, -1])
    net['pc_fc1'] = tf_util.fully_connected(net['pc_maxpool'], 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net['pc_dp1'] = tf_util.dropout(net['pc_fc1'], keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net['pc_fc2'] = tf_util.fully_connected(net['pc_dp1'], 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net['pc_dp2'] = tf_util.dropout(net['pc_fc2'], keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net['pc_fc3'] = tf_util.fully_connected(net['pc_dp2'], 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_model_add_global(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    net = {}
    net['pc_input'] = point_cloud

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    end_points['transform_3'] = transform
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    input_image = tf.expand_dims(point_cloud_transformed, -1)
    net['pc_transformed_3'] = input_image

    net['pc_conv1'] = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net['pc_conv2'] = tf_util.conv2d(net['pc_conv1'], 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net['pc_conv2'], is_training, bn_decay, K=64)
    end_points['transform_64'] = transform
    net_transformed = tf.matmul(tf.squeeze(net['pc_conv2'], axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])
    net['pc_transformed_64'] = net_transformed

    net['pc_conv3'] = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net['pc_conv4'] = tf_util.conv2d(net['pc_conv3'], 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net['pc_conv5'] = tf_util.conv2d(net['pc_conv4'], 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    net['pc_conv6'] = tf_util.conv2d(net['pc_conv5'], 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    
    net['pc_conv7'] = tf_util.conv2d(net['pc_conv6'], 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)

    # Symmetric function: max pooling
    global_feat = tf_util.max_pool2d(net['pc_conv7'], [num_point,1],
                             padding='VALID', scope='maxpool')

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])

    add_feat = tf.add(net['pc_conv5'], global_feat_expand, name = 'add')

    net['pc_conv8'] = tf_util.conv2d(add_feat, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)

    net['pc_maxpool'] = tf_util.max_pool2d(net['pc_conv8'], [num_point,1],
                             padding='VALID', scope='maxpool')

    net['pc_maxpool'] = tf.reshape(net['pc_maxpool'], [batch_size, -1])
    net['pc_fc1'] = tf_util.fully_connected(net['pc_maxpool'], 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net['pc_dp1'] = tf_util.dropout(net['pc_fc1'], keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net['pc_fc2'] = tf_util.fully_connected(net['pc_dp1'], 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net['pc_dp2'] = tf_util.dropout(net['pc_fc2'], keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net['pc_fc3'] = tf_util.fully_connected(net['pc_dp2'], 40, activation_fn=None, scope='fc3')

    return net, end_points



def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform_64'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
