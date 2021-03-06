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
from tesseract.tessact_recog import text_read
from ctpn.main.ctpn_pred import ctpn_pred
from shapedetect.shapedetect import  inoutlet_detect,square_detect,circle_detect
from linedetect.linedetect import houghline_detect
from maskall.maskall import maskall
from compmatch.compmatch import compmatch
from graph.pidgraph import create_graph
from preprocess.preprocess import get_images

tf.app.flags.DEFINE_string('test_data_path', '../Data/demo-pid-sim', '')
tf.app.flags.DEFINE_string('removeborder_output_path', 'preprocess/data/res_border', '')

tf.app.flags.DEFINE_string('morpho_input_path', 'maskall/data/res', '')
tf.app.flags.DEFINE_string('morpho_output_path', 'preprocess/data/res_morpho', '')


# shape detection directory
tf.app.flags.DEFINE_string('inoutletdetect_input_path', '../Data/demo-pid-sim', '')
tf.app.flags.DEFINE_string('inoutletdetect_output_path', 'shapedetect/data/res_inoutlet', '')

tf.app.flags.DEFINE_string('squaredetect_input_path', '../Data/demo-pid-sim', '')
tf.app.flags.DEFINE_string('squaredetect_output_path', 'shapedetect/data/res_square', '')

tf.app.flags.DEFINE_string('circledetect_input_path', '../Data/demo-pid-sim', '')
tf.app.flags.DEFINE_string('circledetect_output_path', 'shapedetect/data/res_circle', '')

tf.app.flags.DEFINE_string('maskloc_output_path', 'shapedetect/data', '')

# ctpn text detection directory
tf.app.flags.DEFINE_string('ctpn_input_path', '../Data/demo-pid-sim', '')
tf.app.flags.DEFINE_string('ctpn_output_path', 'ctpn/data/res', '')
tf.app.flags.DEFINE_string('textloc_output_path', 'ctpn/data', '')

# text recognition directory
tf.app.flags.DEFINE_string('tessact_input_path', 'ctpn/data/res', '')
tf.app.flags.DEFINE_string('tessact_output_path', 'tesseract/data/res', '')

# line detection directory
tf.app.flags.DEFINE_string('pipedetect_input_path', 'maskall/data/res', '')
tf.app.flags.DEFINE_string('pipedetect_output_path', 'linedetect/data/res_pipe', '')
tf.app.flags.DEFINE_string('lineloc_output_path', 'linedetect/data', '')

# Component match directory
tf.app.flags.DEFINE_string('compmatch_input_path', '../Data/demo-pid-sim', '')
tf.app.flags.DEFINE_string('compmatch_target_path', 'compmatch/data/template', '')
tf.app.flags.DEFINE_string('compmatch_output_path', 'compmatch/data/res_comp', '')
tf.app.flags.DEFINE_string('comploc_output_path', 'compmatch/data', '')
# Mask directory
tf.app.flags.DEFINE_string('maskimage_input_path', '../Data/demo-pid-sim', '')
tf.app.flags.DEFINE_string('shapeloc_input_path', 'shapedetect/data', '')
tf.app.flags.DEFINE_string('txtloc_input_path', 'ctpn/data', '')
tf.app.flags.DEFINE_string('comploc_input_path', 'compmatch/data', '')
tf.app.flags.DEFINE_string('mask_output_path', 'maskall/data/res', '')


tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../Checkpoints_ctpn/', '')
FLAGS = tf.app.flags.FLAGS






def main(argv=None):
    ##remove_border(FLAGS.test_data_path,FLAGS.removeborder_output_path)
    ##skeleton(FLAGS.morpho_input_path,FLAGS.morpho_output_path)
    #inoutlet_detect(FLAGS.inoutletdetect_input_path, FLAGS.inoutletdetect_output_path, True, FLAGS.maskloc_output_path)
    #circle_detect(FLAGS.circledetect_input_path, FLAGS.circledetect_output_path, False, FLAGS.maskloc_output_path)
    ##square_detect(FLAGS.squaredetect_input_path, FLAGS.squaredetect_output_path, False,  FLAGS.maskloc_output_path)
    #ctpn_pred(FLAGS.ctpn_input_path, FLAGS.ctpn_output_path,FLAGS.textloc_output_path, FLAGS.checkpoint_path, FLAGS.gpu)
    ##text_read(FLAGS.tessact_input_path,FLAGS.tessact_output_path)
    #compmatch(FLAGS.compmatch_input_path,FLAGS.compmatch_target_path,FLAGS.compmatch_output_path, FLAGS.comploc_output_path)
    #maskall(FLAGS.maskimage_input_path,FLAGS.shapeloc_input_path,FLAGS.txtloc_input_path, FLAGS.comploc_input_path,FLAGS.mask_output_path)
    
    #houghline_detect(FLAGS.pipedetect_input_path, FLAGS.pipedetect_output_path,  FLAGS.lineloc_output_path)
    img_fn_list = get_images(FLAGS.test_data_path)
    create_graph(img_fn_list, FLAGS.shapeloc_input_path, FLAGS.comploc_output_path,FLAGS.textloc_output_path, FLAGS.lineloc_output_path)
    
if __name__ == '__main__':
    tf.app.run()
