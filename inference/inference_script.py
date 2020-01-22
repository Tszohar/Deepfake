import glob
import os
import shutil
import sys
import zipfile

import pandas as pd

from inference import inf_config
from inference.video_handler import VideoHandler


def extract_zip(zip_file_path: str, dst_folder: str):
    """
    :param zip_file_path: file path to a zip file to be extracted to dst_folder
    :param dst_folder: destination folder for the unzipped files
    :return: This function receives a zip file and extract it to dst_folder
    """

    with zipfile.ZipFile(zip_file_path) as zip:
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)
        print('Extracting zip')
        zip.extractall(dst_folder)
        print('Extraction Completed')


if __name__ == "__main__":
    print(sys.argv)
    videos_folder = sys.argv[1]
    # videos_folder = os.path.splitext(zip_file_path)[0]
    # extract_zip(zip_file_path=zip_file_path, dst_folder=videos_folder)
    videos_file_list = glob.glob1(videos_folder, "*.mp4")
    video_handler = VideoHandler(image_size=inf_config.image_size, frame_decimation=inf_config.frame_decimation,
                                 output_path=inf_config.output_path)
    submission_results = {}
    for video_file_name in videos_file_list:
        video_file_path = os.path.join(videos_folder, video_file_name)
        result = video_handler.handle(video_file_path=video_file_path)
        submission_results.update({(video_file_name, result)})
        print('{} FAKE probability is: {}'.format(video_file_name, result))
    # submission_path = os.path.join(os.path.dirname(zip_file_path), 'submission.csv')
    dataframe = pd.DataFrame(list(submission_results.items()), columns=['filename', 'label'])
    dataframe.to_csv('submission.csv')


