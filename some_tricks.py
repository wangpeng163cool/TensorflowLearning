'''
Autor: Wang Peng
Description: 
Version: 1.0
Date: 2020-11-20 09:27:55
LastEditors: Wang Peng
LastEditTime: 2020-11-20 09:28:03
FilePath: /Workspace/TensorflowLearning/some_tricks.py
'''

# encoding=utf-8
# 本文件用于保存tensorflow 相关技巧代码




import tensorflow as tf
import numpy as np
def get_all_tensor_name():
    # 获取模型所有node对应name

    import os
    from tensorflow.python import pywrap_tensorflow

    checkpoint_path = os.path.join('train', "model.ckpt-239000")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


def freeze_model():
    # 将ckpt模型打包成pb

    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.tools import optimize_for_inference_lib
    from tensorflow.python.framework import meta_graph

    saved_graph_name = 'train/graph.pbtxt'
    saved_ckpt_name = 'train/model.ckpt-239000'
    out_node_name = 'model_output'

    output_frozen_graph_name = 'train/frozen_nmt.pb'

    freeze_graph.freeze_graph(input_graph=saved_graph_name, input_saver='',
                              input_binary=False, input_checkpoint=saved_ckpt_name, output_node_names='model_output',
                              restore_op_name='', filename_tensor_name='',
                              output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes='')


def embedding_lookup():

    # 帮助理解embedding_lookup
    # 就是根据id检索信息
    # 实现word_embedding

    a = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [
        2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
    a = np.asarray(a)

    idx1 = tf.Variable([0, 4, 3, 1], tf.int32)
    out1 = tf.nn.embedding_lookup(a, idx1)

    idx2 = tf.Variable([[0, 2, 3, 1], [4, 2, 1, 3]], tf.int32)
    out2 = tf.nn.embedding_lookup(a, idx2)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(out1))

        print('--------')
        print(sess.run(out2))


def tensorf_float_and_int_convert():

    # tensor 类型转换
    # tf.round() 四舍五入

    a = tf.constant([1.2, 3.8, 5.9], dtype=tf.float32)
    b = tf.cast(tf.round(a), dtype=tf.int32)

    with tf.Session() as sess:
        c = sess.run(b)
        print(c)


def shape_list(x):
    '''
    @author: Wang Peng
    @description: return shape of x no matter x is static or dynamic
    @param {type} 
    @return: 
    '''
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    re_ans = [dynamic[i] if s in None else s for i, s in enumerate(static)]
    return re_ans


if __name__ == '__main__':
    tensorf_float_and_int_convert()
