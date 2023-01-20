from concurrent.futures import ThreadPoolExecutor
from goprocam import GoProCamera
from datetime import datetime
import processing as pp
import argparse
import json
import os

# Default parameters
DEFAULT_OUTPUT_DIR = 'output_videos'
DEFAULT_RECORD_DURATION = 15
DEFAULT_FPS = 40
DEFAULT_CONFIG_FILE = 'config.json'
DEFAULT_VIDEO_FILE_SIZE_MB = 13
DEFAULT_VIDEO_FILE_SIZE_MARGIN_PERCENT = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utopia - Base 360 con GoPro')
    parser.add_argument('-f', '--fps',             type=int, default=DEFAULT_FPS)
    parser.add_argument('-o', '--output_dir',      type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('-c', '--config_file',     type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument('-r', '--record_duration', type=int, default=DEFAULT_RECORD_DURATION)
    parser.add_argument('-s', '--file_size',       type=int, default=DEFAULT_VIDEO_FILE_SIZE_MB)
    parser.add_argument('-m', '--margin_percent',  type=int, default=DEFAULT_VIDEO_FILE_SIZE_MARGIN_PERCENT)
    args = parser.parse_args()
    
    # Check if output directory exists, and if not, try to create it
    if not os.path.exists(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except:
            raise RuntimeError('Specified output directory does not exist and could not be created')
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        raise RuntimeError('Specified configuration file does not exist')
    
    # Read config file
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Initialize variables and main loop
    gopro = GoProCamera.GoPro()
    print('')
    recording = False
    processing = []
    executor = ThreadPoolExecutor(max_workers=1)
    while True:
        msg = input('Ingrese el nombre de la carpeta para grabar o Q para salir: ')
        if msg == 'q':
            break
        else:
            print('Grabando...')
            video_url = gopro.shoot_video(args.record_duration)
            print('Descargando video...')
            default_folder_name = str(datetime.now()).replace(' ','__').split('.')[0].replace(':','-')
            if len(msg) > 0:
                using_default_folder_name = False
                folder_name = msg
            else:
                using_default_folder_name = True
                folder_name = default_folder_name
            folder_path = args.output_dir.rstrip('/') + '/' + folder_name
            while os.path.exists(folder_path):
                if '(' in folder_name and ')' in folder_name:
                    number = int(folder_name.split('(')[-1].split(')')[0]) + 1
                else:
                    number = 1
                folder_name = folder_name.split('(')[0].rstrip(' ') + ' (' + str(number) + ')'
                folder_path = args.output_dir.rstrip('/') + '/' + folder_name
            else:
                try:
                    os.mkdir(folder_path)
                except:
                    if not using_default_folder_name:
                        folder_path = args.output_dir.rstrip('/') + '/' + default_folder_name
                        os.mkdir(folder_path)
                    else:
                        raise RuntimeError('Unable to create a folder to store the output videos')
            
            file_path = pp.download_video(video_url, folder_path)
            recording = False
        
            # Send video to processing
            future = executor.submit(pp.process_video, file_path, folder_path, args.fps, config, 
                                     args.file_size, args.margin_percent)
            processing.append((future, file_path))
            
            # Remove processed videos from list
            to_remove = []
            for i, (future, file_path) in enumerate(processing):
                if future.done():
                    to_remove.append((i, file_path))
            if len(to_remove) > 0:
                removed = 0
                for i, file_path in to_remove:
                    processing.pop(i-removed)
                    removed += 1
