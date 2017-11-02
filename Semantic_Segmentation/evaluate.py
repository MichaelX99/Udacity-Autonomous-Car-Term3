import tensorflow as tf
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import scipy.misc

def segment(img, sess, input_tensor, keep_prob, segment_op):

    img_shape = (160, 576)
    img = scipy.misc.imresize(img, img_shape)

    im_softmax = sess.run([segment_op], {keep_prob: 1.0, input_tensor: [img]})

    im_softmax = im_softmax[0][:, 1].reshape(img_shape[0], img_shape[1])
    segmentation = (im_softmax > 0.5).reshape(img_shape[0], img_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(img)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)

def run():
    #path = './runs/frozen.pb'
    path = './runs/optimized.pb'
    #path = './runs/eightbit_graph.pb'

    segmentation_graph = tf.Graph()
    with segmentation_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(seg_graph_def, name='')


        input_tensor = tf.get_default_graph().get_tensor_by_name('input_image:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        logits = tf.get_default_graph().get_tensor_by_name('logits:0')

        segment_op = tf.nn.softmax(logits)

        with tf.Session() as sess:
            project_output = './runs/annotated_project.mp4'
            project_clip = VideoFileClip("project_video.mp4")
            project_out = project_clip.fl_image(lambda x: segment(x, sess, input_tensor, keep_prob, segment_op))
            project_out.write_videofile(project_output, audio=False)


if __name__ == '__main__':
    run()
