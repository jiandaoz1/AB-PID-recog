# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

cwd = os.getcwd()
cwd="D:\Abyss_Project\Code"
sys.path.append(cwd)
os.chdir(cwd) 
print(os.getcwd())

from preprocess.preprocess import remove_border, skeleton,thinning
from preprocess.preprocess import get_images
from preprocess.preprocess import resize_image
from tesseract.tessact_recog import text_read
from ctpn.nets import model_train as model
from ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from ctpn.utils.text_connector.detectors import TextDetector
from lineshapedetect.lineshapedetect import houghline_detect, inoutlet_detect,square_detect,circle_detect

tf.app.flags.DEFINE_string('test_data_path', '../Data/demo-pid-red', '')
tf.app.flags.DEFINE_string('removeborder_output_path', 'preprocess/data/res_border', '')

tf.app.flags.DEFINE_string('morpho_input_path', 'preprocess/data/res_border', '')
tf.app.flags.DEFINE_string('morpho_output_path', 'preprocess/data/res_morpho', '')

tf.app.flags.DEFINE_string('inoutletdetect_input_path', 'preprocess/data/res_border', '')
tf.app.flags.DEFINE_string('inoutletdetect_output_path', 'lineshapedetect/data/res_inoutlet', '')

tf.app.flags.DEFINE_string('squaredetect_input_path', 'preprocess/data/res_border', '')
tf.app.flags.DEFINE_string('squaredetect_output_path', 'lineshapedetect/data/res_square', '')

tf.app.flags.DEFINE_string('circledetect_input_path', 'preprocess/data/res_border', '')
tf.app.flags.DEFINE_string('circledetect_output_path', 'lineshapedetect/data/res_circle', '')

tf.app.flags.DEFINE_string('ctpn_input_path', 'preprocess/data/res_morpho', '')
tf.app.flags.DEFINE_string('ctpn_output_path', 'ctpn/data/res', '')

tf.app.flags.DEFINE_string('tessact_input_path', 'ctpn/data/res', '')
tf.app.flags.DEFINE_string('tessact_output_path', 'tesseract/data/res', '')

#tf.app.flags.DEFINE_string('shapemask_input_path', 'preprocess/data/res', '')
#tf.app.flags.DEFINE_string('shapemask_output_path', 'textshapemask/data/res_shape', '')


tf.app.flags.DEFINE_string('pipedetect_input_path', 'preprocess/data/res_morpho', '')
tf.app.flags.DEFINE_string('pipedetect_output_path', 'lineshapedetect/data/res_pipe', '')



tf.app.flags.DEFINE_string('output_path', '../Data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../Checkpoints_ctpn/', '')
FLAGS = tf.app.flags.FLAGS



def ctpn():
    if os.path.exists(FLAGS.ctpn_output_path):
        shutil.rmtree(FLAGS.ctpn_output_path)
    os.makedirs(FLAGS.ctpn_output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():
        
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images(FLAGS.ctpn_input_path)
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    img_raw = cv2.imread(im_fn)
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue
                # image used to draw bounding box
                img_draw = img_raw.copy()

                img, (rh, rw) = resize_image(img_raw)
                h, w, c = img.shape
                for ifrot in ['orig','rot']:
                    im = img.copy()

                    if ifrot == 'rot':
                        im = cv2.transpose(im)
                        im = cv2.flip(im,1)
                        bbox_color = (255,0,0)
                        im_info = np.array([w, h, c]).reshape([1, 3])
                    else: 
                        bbox_color = (0,255,0)
                        im_info = np.array([h, w, c]).reshape([1, 3])
                    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                           feed_dict={input_image: [im],
                                                                      input_im_info: im_info})
    
                    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                    scores = textsegs[:, 0]
                    textsegs = textsegs[:, 1:5]
                    
                    textdetector = TextDetector(DETECT_MODE='H')
                    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], im.shape[:2])
                    boxes = np.array(boxes, dtype=np.int)
                    print(len(boxes))
                    cost_time = (time.time() - start)
                    print("cost time: {:.2f}s".format(cost_time))
                    fx=1.0 / rw
                    fy=1.0 / rh
                    for i, box in enumerate(boxes):
                        if ifrot == 'rot':
                            box = np.array([box[3],h-box[2],box[5],h-box[4],box[7],h-box[6],box[1],h-box[0],box[8]])
                        #resize the images
                        box[:8:2] = (box[:8:2]*fx).astype(np.int32)
                        box[1::2] = (box[1::2]*fy).astype(np.int32)
                        
                        cv2.polylines(img_draw, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=bbox_color, thickness=2)
                        # crop image with rectangle box and save
                        x0,y0,w0,h0 = cv2.boundingRect(box[:8].astype(np.int32).reshape((-1, 2)))
                        img_crop = img_raw[y0:y0+h0,x0:x0+w0].copy()

                        cv2.imwrite(os.path.join(FLAGS.ctpn_output_path, ifrot+str(format(i, "04"))+"-"+os.path.basename(im_fn)), img_crop) 
                        cv2.putText(img_draw, str(i), (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX ,1.0, bbox_color, 2, cv2.LINE_AA) 
   
                    #im = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(FLAGS.ctpn_output_path, ifrot+"-"+os.path.basename(im_fn)),img_draw[:, :, ::-1])

                    with open(os.path.join(FLAGS.ctpn_output_path, ifrot+"-"+os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                                "a") as f:
                        for i, box in enumerate(boxes):
                            line = ",".join(str(box[k]) for k in range(8))
                            line += "," + str(scores[i]) + "\r\n"
                            f.writelines(line)
                        f.close()




def main(argv=None):
    #remove_border(FLAGS.test_data_path,FLAGS.removeborder_output_path)
    #skeleton(FLAGS.morpho_input_path,FLAGS.morpho_output_path)
    inoutlet_detect(FLAGS.inoutletdetect_input_path, FLAGS.inoutletdetect_output_path, mask = False)
    circle_detect(FLAGS.circledetect_input_path, FLAGS.circledetect_output_path, mask = False)
    square_detect(FLAGS.squaredetect_input_path, FLAGS.squaredetect_output_path)
    houghline_detect(FLAGS.pipedetect_input_path, FLAGS.pipedetect_output_path)
    #ctpn()
    #text_read(FLAGS.tessact_input_path,FLAGS.tessact_output_path)
    #shape_mask(FLAGS.shapemask_input_path,FLAGS.shapemask_output_path)
    #inoutlet_detect(FLAGS.inoutletdetect_input_path, FLAGS.inoutletdetect_output_path)
    
    

if __name__ == '__main__':
    tf.app.run()
