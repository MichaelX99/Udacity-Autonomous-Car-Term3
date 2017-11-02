import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import tensorflow.contrib.slim as slim

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    t_in = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    t_kp = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    t_4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    t_3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    t_7 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return t_in, t_kp, t_3, t_4, t_7
tests.test_load_vgg(load_vgg, tf)

def vgg15(input, keep_prob):
    #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py
    layer1 = slim.repeat(input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    layer1 = slim.max_pool2d(layer1, [2, 2], scope='pool1')
    layer2 = slim.repeat(layer1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    layer2 = slim.max_pool2d(layer2, [2, 2], scope='pool2')
    layer3 = slim.repeat(layer2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    layer3 = slim.max_pool2d(layer3, [2, 2], scope='pool3')
    layer4 = slim.repeat(layer3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    layer4 = slim.max_pool2d(layer4, [2, 2], scope='pool4')
    layer5 = slim.repeat(layer4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    layer5 = slim.max_pool2d(layer5, [2, 2], scope='pool5')

    # Use conv2d instead of fully_connected layers.
    layer6 = slim.conv2d(layer5, 4096, [7, 7], scope='fc6')
    layer6 = slim.dropout(layer6, keep_prob, scope='dropout6')
    layer7 = slim.conv2d(layer6, 4096, [1, 1], scope='fc7')
    layer7 = slim.dropout(layer7, keep_prob, scope='dropout7')

    return layer3, layer4, layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], num_outputs = num_classes,
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        weights_initializer=tf.random_normal_initializer(stddev=0.01),
                        activation_fn=None):
        layer7a_out = slim.conv2d(vgg_layer7_out, kernel_size=[1,1])

        # upsample
        layer4a_in1 = slim.conv2d_transpose(layer7a_out, kernel_size=[4,4], stride=2)

        # make sure the shapes are the same!
        # 1x1 convolution of vgg layer 4
        layer4a_in2 = slim.conv2d(vgg_layer4_out, kernel_size=[1,1])

        # skip connection (element-wise addition)
        layer4a_out = tf.add(layer4a_in1, layer4a_in2)

        # upsample
        layer3a_in1 = slim.conv2d_transpose(layer4a_out, kernel_size=[4,4], stride=2)

        # 1x1 convolution of vgg layer 3
        layer3a_in2 = slim.conv2d(vgg_layer3_out, kernel_size=[1,1])

        # skip connection (element-wise addition)
        layer3a_out = tf.add(layer3a_in1, layer3a_in2)

        # upsample
        nn_last_layer = slim.conv2d_transpose(layer3a_out, kernel_size=[16,16], stride=8, scope="nn_last_layer")

    return nn_last_layer

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))

    # define training operation
    momentum = .9
    #optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
            print("Loss: = {:.3f}".format(loss))
        print()
tests.test_train_nn(train_nn)

def extract_weights(sess, vgg_path):
    #https://github.com/asimonov/CarND3-P2-FCN-Semantic-Segmentation/blob/master/main.py
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    filtered_variables = [op for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name == 'VariableV2']

    var_values = {}
    for var in filtered_variables:
        name = var.name
        tensor = tf.get_default_graph().get_tensor_by_name(name + ':0')
        value = sess.run(tensor)
        name = name.replace('filter', 'weights')
        if name[:4] == "conv":
            name = name[:5] + '/' + name
        var_values[name] = value

    return var_values

def assign_vgg(sess, vgg_weights):
    #https://github.com/asimonov/CarND3-P2-FCN-Semantic-Segmentation/blob/master/fcn8vgg16.py
    tensors = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    for temp in tensors:
        key_name = temp.name[:-2]
        if key_name in vgg_weights.keys():
            sess.run(temp.assign(vgg_weights[key_name]))


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    vgg_path = os.path.join(data_dir, 'vgg')

    VGG_graph = tf.Graph()
    Seg_graph = tf.Graph()

    with tf.Session(graph=VGG_graph) as sess:
        vgg_weights = extract_weights(sess, vgg_path)

    with tf.Session(graph=Seg_graph) as sess:
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = vgg15(input_image, keep_prob)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())

        assign_vgg(sess, vgg_weights)

        # TODO: Train NN using the train_nn function
        epochs = 50
        batch_size = 3
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        logits = tf.identity(logits, "logits")

        saver = tf.train.Saver()

        # TODO: Save inference data using helper.save_inference_samples
        print("Saving Model")
        saver.save(sess, runs_dir + '/model.ckpt')
        tf.train.write_graph(sess.graph_def, runs_dir, 'graph_def.pb', False)
        print("Saved model.  Now starting evaluation")

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        MODEL_NAME = 'seg'
        input_graph_path = 'runs/graph_def.pb'
        checkpoint_path = 'runs/model.ckpt'
        input_saver_def_path = ""
        input_binary = True
        output_node_names = "logits"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = 'runs/frozen.pb'
        clear_devices = True
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



if __name__ == '__main__':
    run()
