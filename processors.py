import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from glob import glob
import IPython.display as ipd
from tqdm.notebook import tqdm
from pathlib import Path
from PIL import Image
import subprocess
import os
import pytesseract
import array
import easyocr

text_results_list = np.array([])
confidence_results_list = np.array([])
reader = easyocr.Reader(['en'], gpu = True)


class VideoProcessor:
    def __init__(self, file_name, file_type, folder_path, fps, output_image_type, frames_path):
        self.file_name = file_name
        self.file_type = file_type
        self.folder_path = folder_path
        self.fps = fps
        self.output_image_type = output_image_type
        self.file_path = folder_path + file_name + file_type
        self.frames_path = frames_path
        
    
    def convert_to_mp4(self):
        if self.file_path.lower().endswith('.mov'):
            subprocess.run(['ffmpeg', '-i', self.file_path, '-qscale', '0', self.file_name + '.mp4'])

    def extract_frames(self):
        x = self.folder_path + self.file_name + '_' + self.fps + '_frames'
        if not os.path.exists(x):
            os.makedirs(x)
        subprocess.run(['ffmpeg', '-i', self.file_path, '-vf', 'fps=' + self.fps, x + '/%d' + self.output_image_type])

    def extract_text_EasyOCR(self):
        global text_results_list
        line_counter = 0
        error_counter = 0
        error_line_number = []
        files = Path(self.frames_path).glob('*'+ self.output_image_type)
        for file in files:
            line_counter = line_counter + 1
            results = reader.readtext(self.frames_path + "/"+str(line_counter)+self.output_image_type)
            text = ' '.join([text for _, text, _ in results])
            details = pd.DataFrame(results, columns=['bbox','text','conf'])
            print(details)

            if text.strip().isdigit():
                int(text)
                text_results_list = np.append(text_results_list, text)
            else:
                error_counter = error_counter + 1
                error_line_number.append(line_counter)
                text_results_list = np.append(text_results_list, np.nan)

        with open(self.frames_path +'/test.txt', 'w') as file:
            for element in text_results_list: file.write(f"{element}\n")
            file.write(f"{error_counter} at lines: ")
            for line in error_line_number: file.write(f"{line}, ")

    def extract_text_tesseract(self):
        global text_results_list
        global confidence_results_list

        line_counter = 0
        error_counter = 0
        error_line_number = []
        files = Path(self.frames_path).glob('*'+ self.output_image_type)
        for file in files:
            line_counter = line_counter + 1

            
            image_path = self.frames_path + "/"+str(line_counter) + self.output_image_type
            image = Image.open(image_path)
            text = pytesseract.image_to_data(image) #, output_type=pytesseract.Output.DICT
            # text = data['text'][0] 
            # confidence = data['conf'][0] 

            # confidence_results_list = np.append(confidence_results_list, confidence)
            # if text.strip().isdigit():
                # int(text)
            text_results_list = np.append(text_results_list, text)
            # else:
            #     error_counter = error_counter + 1
            #     error_line_number.append(line_counter)
            #     text_results_list = np.append(text_results_list, np.nan)

        with open(self.frames_path +'/test.txt', 'w') as file:
            for element in text_results_list:
                file.write(f"{element}\n")
            # for element, confidence in zip(text_results_list, confidence_results_list):
            #     file.write(f"{element} (Confidence: {confidence})\n")


            file.write(f"{error_counter} at lines: ")
            for line in error_line_number: file.write(f"{line}, ")
    def preprocess_image_EasyOCR(self):
        counter = 0
        files = Path(self.frames_path).glob('*'+ self.output_image_type)
        
        for file in files:
            counter = counter + 1

            img = cv2.imread(self.frames_path + '/' + str(counter) + self.output_image_type, cv2.IMREAD_GRAYSCALE)
            img = cv2.fastNlMeansDenoising(img, None, 9, 7, 31)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, 8)
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img = cv2.dilate(img, kernel, iterations=1)
            img = cv2.erode(img, kernel, iterations=2)
            # use this for  pytesseract img = cv2.GaussianBlur(img, (29, 29),0) #adjust this value 29 if you want the best result
            width = int(img.shape[1] * 0.57)  # Reduce width by 0.57%
            height = int(img.shape[0] * 0.5)  # Reduce height by 50%
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)



            cv2.imwrite(self.frames_path + '/' + str(counter) + self.output_image_type, img)
                
    def preprocess_image_tesseract(self):
        counter = 0
        files = Path(self.frames_path).glob('*'+ self.output_image_type)
        
        for file in files:
            counter = counter + 1

            img = cv2.imread(self.frames_path + '/' + str(counter) + self.output_image_type, cv2.IMREAD_GRAYSCALE)
            img = cv2.fastNlMeansDenoising(img, None, 9, 7, 31)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, 8)
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img = cv2.dilate(img, kernel, iterations=1)
            img = cv2.erode(img, kernel, iterations=2)
            img = cv2.GaussianBlur(img, (29, 29),0) #adjust this value 29 if you want the best result
            width = int(img.shape[1] * 0.57)  # Reduce width by 0.57%
            height = int(img.shape[0] * 0.5)  # Reduce height by 50%
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imwrite(self.frames_path + '/' + str(counter) + self.output_image_type, img)
