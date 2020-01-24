import glob
import os
import sys

import pandas as pd

from inference import inf_config
from inference.video_handler import VideoHandler


def run(base_dir: str):
    videos_file_list = glob.glob1(base_dir, "*.mp4")
    video_handler = VideoHandler(image_size=inf_config.image_size,
                                 frame_decimation=inf_config.frame_decimation,
                                 output_path=inf_config.output_path,
                                 model_path=inf_config.model)
    submission_results = {}
    for video_file_name in videos_file_list:
        video_file_path = os.path.join(base_dir, video_file_name)
        result = video_handler.handle(video_file_path=video_file_path)
        submission_results.update({(video_file_name, result)})
        print('{} FAKE probability is: {}'.format(video_file_name, result))
    dataframe = pd.DataFrame(list(submission_results.items()), columns=['filename', 'label'])
    dataframe.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    videos_folder = sys.argv[1]
    run(base_dir=videos_folder)
