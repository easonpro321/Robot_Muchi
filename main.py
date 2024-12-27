from processors import VideoProcessor

file_name = 'CMPT1'
file_type = '.mp4'
folder_path = 'C:/Users/eason/Documents/All/projects/3DPrints/Robot_Muchi/Trails/Legs/Top/V1/Motor/'
fps = '0.1'
output_image_type = '.png'
frames_path = 'C:/Users/eason/Documents/All/projects/3DPrints/Robot_Muchi/Trails/Legs/Top/V1/Motor/CMPT1_0.1_frames'
video = VideoProcessor(file_name, file_type, folder_path, fps, output_image_type, frames_path)

video.extract_text()