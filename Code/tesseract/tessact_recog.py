import pytesseract
from PIL import Image
import cv2
import numpy as np
from glob import glob
import os
import shutil



def text_read(input_path,output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    print(input_path)
    for ifrot in ['orig','rot']:
        cropimg_fn_list = glob(input_path+"/"+ifrot+('[0-9]' * 4)+"*")
        for cropimg_fn in cropimg_fn_list:
            cropimg = Image.open(cropimg_fn)
            cropimg.save(input_path+"/ocr.png", dpi=(300, 300))
            cropimg = cv2.imread(input_path+"/ocr.png")
            cropimg = cv2.resize(cropimg, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            retval, threshold = cv2.threshold(cropimg,127,255,cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(threshold,lang='eng', config='--psm 6 --oem 2 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-/\\"')



           # cropimg.save(input_path+"/ocr.png", dpi=(300, 300))
           # cropimg = cv2.imread(input_path+"/ocr.png")
           # cropimg = cv2.resize(cropimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
          #  retval, threshold = cv2.threshold(cropimg,127,255,cv2.THRESH_BINARY)
          #  text = pytesseract.image_to_string(threshold,lang='eng')
            cropimg_fn_base = os.path.basename(cropimg_fn)
            cropimg_fn_wo_ext = os.path.splitext(cropimg_fn_base)[0]
            with open(output_path+"/"+cropimg_fn_wo_ext+".txt", "w",5 ,"utf-8") as text_file:
                text_file.write(text)

