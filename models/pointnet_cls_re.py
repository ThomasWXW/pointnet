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

def get_distance(point_cloud):
    batch_size = point_cloud.get_shape()[0].value
    distance_pc = tf.constant([])

    # mean and var
    mean = 0
    var = 0.1

    for ba in range(batch_size):
        pc_x = point_cloud[ba, :, 0]
        pc_y = point_cloud[ba, :, 1]
        pc_z = point_cloud[ba, :, 2]

        pc_x1, pc_x2 = tf.meshgrid(pc_x, pc_x, name='pc_x_mesh')
        pc_y1, pc_y2 = tf.meshgrid(pc_y, pc_y, name='pc_y_mesh')
        pc_z1, pc_z2 = tf.meshgrid(pc_z, pc_z, name='pc_z_mesh')

        diff_x = tf.squared_difference(pc_x1, pc_x2, name='diff_x')
        diff_y = tf.squared_difference(pc_y1, pc_y2, name='diff_y')
        diff_z = tf.squared_difference(pc_z1, pc_z2, name='diff_z')

        distance = tf.sqrt(diff_x + diff_y + diff_z, name='sqrt')

        max_distance = tf.reduce_max(distance, name='reduce_max')

        distance = tf.div(distance, tf.maximum(max_distance, 1e-12), name='div1')

        distance = tf.expand_dims(distance, 0, name='expand_dims')

        exp_pc = tf.exp(-tf.div(tf.square(distance - mean, name='square'), 2* var * var, name='div2'), name='exp')

        reduce_pc = tf.reduce_sum(exp_pc, axis=2, name='reduce_sum')

        _,reduce_pc = tf.meshgrid(reduce_pc, reduce_pc, name='pc_x_mesh')

        distance = tf.div(exp_pc, reduce_pc, name='div3')

        if ba == 0:
            distance_pc = distance
        else:
            distance_pc = tf.concat([distance_pc, distance], axis=0, name='concat')

    return distance_pc

def get_distance_v2(point_cloud):
    with tf.variable_scope('get_distance_v2') as sc:
        point_cloud = tf.squeeze(point_cloud)
        batch_size = point_cloud.get_shape()[0].value
        num_points = point_cloud.get_shape()[1].value
        feature_dim = point_cloud.get_shape()[2].value

        pc_reshaped = tf.reshape(point_cloud, [-1])
        pc_reshaped = tf.reshape(pc_reshaped, [batch_size * num_points, feature_dim])
        pc_reshaped = tf.expand_dims(pc_reshaped, [1])

        pc_xyz1 = tf.tile(pc_reshaped, [1, num_points, 1])
        pc_xyz2 = tf.concat(tf.split(tf.transpose(pc_xyz1, [1, 0, 2]), batch_size, axis=1), 0)

        pc_dist_2 = tf.reduce_sum(tf.squared_difference(pc_xyz1, pc_xyz2), axis=2)
        pc_max_dist_2 = tf.reduce_max(pc_dist_2)

        var = 0.1

        pc_gaussion_w = tf.exp(-1 * tf.div(pc_dist_2, 2 * pc_max_dist_2 * var * var))

        pc_gaussion_w = tf.reshape(pc_gaussion_w, [batch_size, num_points, num_points])
        
    return pc_gaussion_w

def get_distance_v3(point_cloud):
    with tf.variable_scope('get_distance_v3') as sc:
        point_cloud = tf.squeeze(point_cloud)
        batch_size = point_cloud.get_shape()[0].value
        num_points = point_cloud.get_shape()[1].value
        feature_dim = point_cloud.get_shape()[2].value

        point_cloud = tf.expand_dims(point_cloud, axis=2)
        point_cloud = tf.tile(point_cloud, [1, 1, num_points, 1])
        point_cloud_t = tf.transpose(point_cloud, [0, 2, 1, 3])
        point_cloud = tf.reduce_sum(tf.squared_difference(point_cloud, point_cloud_t), axis = 3)
        pc_max_dist = tf.reduce_max(point_cloud)

        var = 0.1

        pc_gaussion_w = tf.exp(-1 * tf.div(point_cloud, 2 * pc_max_dist * var * var))
        
    return pc_gaussion_w

def gaussion_filter(feature, weight):
    with tf.variable_scope('gaussion_filter') as sc:
        feature = tf.squeeze(feature)
        batch_size = feature.get_shape()[0].value
        num_points = feature.get_shape()[1].value
        feature_dim = feature.get_shape()[2].value

        feature = tf.tile(tf.expand_dims(feature, axis = 2),[1, 1, num_points, 1])
        weight = tf.tile(tf.expand_dims(weight, -1), [1, 1, 1, feature_dim])

        feature = tf.reduce_sum(tf.multiply(feature, weight), axis = 2, keep_dims = True)

    return feature

def gaussion_filter_v2(feature, weight):
    with tf.variable_scope('gaussion_filter_v2') as sc:
        feature = tf.squeeze(feature)
        batch_size = feature.get_shape()[0].value
        num_points = feature.get_shape()[1].value
        feature_dim = feature.get_shape()[2].value

        weight = tf.tile(tf.expand_dims(weight, 1), [1, feature_dim, 1, 1])
        feature = tf.transpose(tf.expand_dims(feature, -1), [0, 2, 1, 3])

        pc_filtered = tf.transpose(tf.matmul(weight, feature), [0, 2, 3, 1])

    return pc_filtered

def filter_gaussion(feature, filter):
    with tf.variable_scope('filter_gaussion') as sc:
        feature = tf.squeeze(feature, name='squeeze1')
        batch_size = feature.get_shape()[0].value
        feature_dim = feature.get_shape()[2].value
        num_points = feature.get_shape()[1].value

        new_feature = tf.constant([])
        eye_matrix = tf.eye(1024)

        for ba in range(batch_size):
            cur_filter = filter[ba,:,:]
            new_cur_feature = tf.constant([])

            for dim in range(feature_dim):
                cur_feat = feature[ba, :, dim]
                _, cur_feat = tf.meshgrid(cur_feat, cur_feat, name='meshgrid1')
                data_filtered = tf.multiply(cur_feat, cur_filter, name='mul_ew1')

                data_filtered = tf.multiply(data_filtered, eye_matrix, name='mul_ew2')

                data_res = tf.reduce_sum(data_filtered, axis=1, keep_dims=True, name='reduce_sum1')
                #data_res = tf.expand_dims(data_res, -1, name='expand1')
                #print data_res

                if dim == 0:
                    new_cur_feature = data_res
                else:
                    new_cur_feature = tf.concat([new_cur_feature, data_res], axis=1, name='concat1')

            new_cur_feature = tf.expand_dims(new_cur_feature, axis=0, name='expand2')
            if ba == 0:
                new_feature = new_cur_feature
            else:
                new_feature = tf.concat([new_feature, new_cur_feature], axis=0, name='concat2')

    return tf.expand_dims(new_feature, [2], name='expand')


def get_model_add_gaussion(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    net = {}
    net['pc_input'] = point_cloud

    distance_pc = get_distance_v3(point_cloud)

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

    net['pc_conv1'] = gaussion_filter_v2(net['pc_conv1'], distance_pc)

    net['pc_conv2'] = tf_util.conv2d(net['pc_conv1'], 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    net['pc_conv2'] = gaussion_filter_v2(net['pc_conv2'], distance_pc)

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

    net['pc_conv3'] = gaussion_filter_v2(net['pc_conv3'], distance_pc)

    net['pc_conv4'] = tf_util.conv2d(net['pc_conv3'], 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)

    net['pc_conv4'] = gaussion_filter_v2(net['pc_conv4'], distance_pc)

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
    with tf.device('/gpu:1'):
        #inputs = tf.zeros((32,1024,3))
        #outputs = get_model(inputs, tf.constant(True))
        inputs = tf.random_uniform([2, 4, 3])

        gaussion = get_distance_v3(inputs)

        input_filtered = gaussion_filter_v2(inputs, gaussion)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        out, out1 = sess.run([input_filtered, inputs])

        print "out:", out
        print "out1:", out1
        
