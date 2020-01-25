import numpy as np
import glob
import json
import os
import sys

import pandas as pd

from inference import inf_config
from inference.video_handler import VideoHandler
from sklearn.metrics import log_loss


def run(base_dir: str):
    meta_data_file = os.path.join(base_dir, "metadata.json")
    json_data = None
    if os.path.isfile(meta_data_file):
        with open(meta_data_file) as f:
            json_data = json.load(f)
    videos_file_list = glob.glob1(base_dir, "*.mp4")
    video_handler = VideoHandler(image_size=inf_config.image_size,
                                 frame_decimation=inf_config.frame_decimation,
                                 output_path=inf_config.output_path,
                                 model_path=inf_config.model)
    submission_results = {}
    results = []
    for video_file_name in videos_file_list:
        assert json_data is None or video_file_name in json_data
        video_file_path = os.path.join(base_dir, video_file_name)
        result = video_handler.handle(video_file_path=video_file_path)
        submission_results.update({(video_file_name, result)})
        print('{} FAKE probability is: {}'.format(video_file_name, result))
        if json_data is not None:
            label = json_data[video_file_name]['label']
            print("Ground truth label: {}".format(label))
            results.append((1-result, float(label == "FAKE")))

            results_np = np.array(results)
            if sum(results_np[:, 1] == 0.) > 0 and sum(results_np[:, 1] == 1.) > 0:
                loss = log_loss(results_np[:, 1], results_np[:, 0])
                print("Loss until now: {}".format(loss))
    dataframe = pd.DataFrame(list(submission_results.items()), columns=['filename', 'label'])
    dataframe.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    videos_folder = sys.argv[1]
    run(base_dir=videos_folder)
