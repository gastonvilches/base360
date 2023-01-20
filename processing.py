import os
import cv2
import json
import gopro
import shutil
import argparse
import requests
import numpy as np
from datetime import datetime

def download_video(url, output_dir):
    '''Download video from url and saves it in output_dir. If output_dir does not exists, it is created.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = 'Original.mp4'
    file_path = os.path.normpath(os.path.join(output_dir, file_name))
    while os.path.exists(file_path):
        file_path = file_path.split('.')[0] + '_.mp4'
    response = requests.get(url)
    open(file_path, "wb").write(response.content)
    return file_path

def video_to_frames(input_file, start_frame, end_frame):
    video_reader = cv2.VideoCapture(input_file)
    if (video_reader.isOpened() == False):
        raise RuntimeError('Error opening video file')
    date = str(datetime.now()).split('.')[0].replace(':', '-').replace(' ', '_')
    output_folder = 'frames_temp_' + date
    os.mkdir(output_folder)
    frame_idx = 0
    filenames = {}
    while True:
        ret, frame = video_reader.read()
        if not ret:
            msg = 'Input video: ' + input_file + 'has not enough frames for specified effects'
            msg += ' (' + str(end_frame - frame_idx + 1) + ' frames missing)'
            raise RuntimeError(msg)
        if frame_idx >= start_frame - 1:
            filename = os.path.normpath(os.path.join(output_folder, 'frame_' + str(frame_idx) + '.jpg'))
            cv2.imwrite(filename, frame)
            filenames[frame_idx] = filename
        if frame_idx >= end_frame:
            break
        frame_idx += 1
    video_reader.release()
    return output_folder, filenames

def get_speed_vector(video_settings, fps_input, fps_output):
    effects = video_settings['effects']
    assert effects[-1]['type'] != 'transition', 'Last effect cant be a transition.'
    k = fps_input/fps_output
    speed = []
    for i in range(len(effects)):
        effect = effects[i]
        num_frames = int(round(effect['duration']*fps_output))
        if effect['type'] == 'normal':
            speed.extend([k]*num_frames)
            curr_speed = k
        if effect['type'] == 'change_speed':
            speed.extend([effect['factor']*k]*num_frames)
            curr_speed = effect['factor']*k
        if effect['type'] == 'reverse':
            speed.extend([-effect['factor']*k]*num_frames)
            curr_speed = -effect['factor']*k
        if effect['type'] == 'transition':
            if effects[i+1]['type'] == 'normal':
                end_speed = k
            if effects[i+1]['type'] == 'change_speed':
                end_speed = effects[i+1]['factor']*k
            if effects[i+1]['type'] == 'reverse':
                end_speed = -effects[i+1]['factor']*k
            speed.extend(list(np.linspace(curr_speed, end_speed, num_frames)))
    speed = np.array(speed)
    speed_sum = np.cumsum(speed)
    if speed_sum.min()/fps_input < -video_settings['start']:
        raise ValueError('The video "' + video_settings['video_name'] + '" tries to reach frames before frame 0.')
    if speed_sum.min()/fps_input < 0:
        print('Warning: The video "' + video_settings['video_name'] + '" reaches frames before start time.')
    return speed
        
def create_video(output_folder, video_settings, frames, speed, fps_input, fps_output, 
                 target_file_size_mb, margin_percent, max_iterations):
    
    def _write_video(scale):
        width_ = int(width*np.sqrt(scale)) if scale != 1 else width
        height_ = int(height*np.sqrt(scale)) if scale != 1 else height
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'avc1'), fps_output, (width_, height_))
        idxi, idxo = (0, 0)
        started = False
        while idxo < len(speed):
            if not started and idxi/fps_input >= video_settings['start']:
                started = True
            if started:
                j = int(idxi)
                if idxi != j:
                    k = idxi - j
                    frame_1 = cv2.imread(frames[j]).astype(np.double)
                    frame_2 = cv2.imread(frames[j+1]).astype(np.double)
                    frame = ((1-k)*frame_1 + k*frame_2).astype(np.uint8)
                else:
                    frame = cv2.imread(frames[j])
                frame = crop_to_aspect_ratio(frame, video_settings['aspect_ratio'])
                if scale != 1:
                    frame = cv2.resize(frame, (width_, height_), interpolation=cv2.INTER_LINEAR)
                video_writer.write(frame)
                idxi += speed[idxo]
                idxo += 1
            if not started:
                idxi += 1
        video_writer.release()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_filename = output_folder.rstrip('/') + '/' + video_settings['video_name'] + '.mp4'
    frame_dummy = cv2.imread(list(frames.values())[0])
    frame_dummy = crop_to_aspect_ratio(frame_dummy, video_settings['aspect_ratio'])
    height, width, _ = frame_dummy.shape    
    scale = 1
    _write_video(scale)
    if target_file_size_mb is not None:
        size = os.path.getsize(video_filename)/(1024*1024)
        ratio = size/target_file_size_mb
        file_sizes = [size]
        scale_values = [scale]
        while (ratio > 1 or ratio < (100-margin_percent)/100):
            if len(file_sizes) >= max_iterations:
                print('Warning: maximum iterations reached. The output video did not reach the target size')
                break
            if len(scale_values) < 2:
                scale = (100-margin_percent/2)/100*target_file_size_mb/size
            else:
                X = np.hstack((np.ones((len(file_sizes),))[:,None], np.array(file_sizes)[:,None]))
                y = np.array(scale_values)
                offset, slope = np.linalg.inv(X.T @ X) @ X.T @ y
                scale = (100-margin_percent/2)/100*(offset + slope*target_file_size_mb)
            _write_video(scale)
            size = os.path.getsize(video_filename)/(1024*1024)
            ratio = size/target_file_size_mb
            file_sizes.append(size)
            scale_values.append(scale)
            
def crop_to_aspect_ratio(frame, aspect_ratio):
    w, h = aspect_ratio.split(':')
    aspect_ratio = int(w)/int(h)
    height, width, channels = frame.shape
    aspect_ratio_frame = width/height
    
    if aspect_ratio_frame == aspect_ratio:
        return frame
    if aspect_ratio_frame > aspect_ratio:
        width_crop = int(round((width - height*aspect_ratio)/2))
        return frame[:,width_crop:-width_crop,:]
    height_crop = int(round((height*aspect_ratio - width)/(2*aspect_ratio)))
    return frame[height_crop:-height_crop,:,:]

def get_num_of_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def get_fps(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
    
def process_video(input_file, output_folder, fps_output, settings,
                  target_file_size_mb, margin_percent, max_iterations=5):
    if not os.path.exists(input_file):
        raise ValueError('Input file does not exist')
    frame_count = get_num_of_frames(input_file)
    fps_input = get_fps(input_file)
    speeds = []
    start_frames = []
    end_frames = []
    names = [setting['video_name'] for setting in settings]
    if len(names) != len(set(names)):
        raise ValueError('video_name values must be different from each other')
    for settings_i in settings:
        start_frames.append(int(settings_i['start']*fps_input))
        speed = get_speed_vector(settings_i, fps_input, fps_output)
        speeds.append(speed)
        end_frames.append(start_frames[-1] + int(np.ceil(np.cumsum(speed).max())) + 1)
    start_frame = min(start_frames)
    end_frame = max(end_frames)
    limiting_effect = names[np.argmax(end_frames)]
    if end_frame >= frame_count:
        msg = 'Input video: ' + input_file + 'has not enough frames for specified effects'
        msg += ' (' + str(end_frame - frame_count + 1) + ' frames missing) - (worst case:'
        msg += ' ' + limiting_effect + ')'
        raise RuntimeError(msg)
    frames_dir, frames = video_to_frames(input_file, start_frame, end_frame)
    for speed, settings_i in zip(speeds, settings):
        create_video(output_folder, settings_i, frames, speed, fps_input, fps_output, 
                     target_file_size_mb, margin_percent, max_iterations)
    shutil.rmtree(frames_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video processing module')
    parser.add_argument('-f', '--fps',            type=int, default=gopro.DEFAULT_FPS)
    parser.add_argument('-i', '--input_dir',      type=str, default=gopro.DEFAULT_OUTPUT_DIR)
    parser.add_argument('-c', '--config_file',    type=str, default=gopro.DEFAULT_CONFIG_FILE)
    parser.add_argument('-s', '--file_size',      type=int, default=gopro.DEFAULT_VIDEO_FILE_SIZE_MB)
    parser.add_argument('-m', '--margin_percent', type=int, default=gopro.DEFAULT_VIDEO_FILE_SIZE_MARGIN_PERCENT)
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        
    folders_and_files = []
    for folder_name in os.listdir(args.input_dir):
        folder_path = args.input_dir.rstrip('/') + '/' + folder_name
        for file_name in os.listdir(folder_path):
            folders_and_files.append((folder_path, folder_path + '/' + file_name))
    for i, (folder, file) in enumerate(folders_and_files):
        print('Processing file', i+1, '/', len(folders_and_files))
        process_video(file, folder, args.fps, config, target_file_size_mb=args.file_size,
                      margin_percent=args.margin_percent)