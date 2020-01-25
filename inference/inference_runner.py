import numpy as np
import glob
import json
import os
import sys

import pandas as pd

from inference import inf_config
from inference.video_handler import VideoHandler
from sklearn.metrics import log_loss


def run(base_dir: str, load_gt_data: bool = False):
    if load_gt_data:
        meta_data_file = os.path.join(base_dir, "metadata.json")
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
        video_file_path = os.path.join(base_dir, video_file_name)
        result = video_handler.handle(video_file_path=video_file_path)
        submission_results.update({(video_file_name, result)})
        print('{} FAKE probability is: {}'.format(video_file_name, result))
        if load_gt_data and json_data is not None and video_file_name in json_data:
            label = json_data[video_file_name]['label']
            print("Ground truth label: {}".format(label))
            results.append((result, float(label == "FAKE")))

            results_np = np.array(results)
            loss = log_loss(y_true=results_np[:, 1], y_pred=results_np[:, 0], labels=[0., 1.])
            print("Loss until now: {}".format(loss))
    dataframe = pd.DataFrame(list(submission_results.items()), columns=['filename', 'label'])
    dataframe.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    videos_folder = sys.argv[1]
    run(base_dir=videos_folder, load_gt_data=True)
