from inference import inf_config
from inference.video_handler import VideoHandler

video_handler = VideoHandler(image_size=inf_config.image_size,
                             frame_decimation=1,
                             output_path=inf_config.output_path,
                             model_path=inf_config.model)
video_file_path = "/media/guy/Files 3/kaggle_competitions/deepfake/dfdc_train_part_0/qyqufaskjs.mp4"
frame_list = video_handler.video_pre_processor.convert_to_frames(
    video_file_path=video_file_path,
    frame_decimation=video_handler._frame_decimation,
    preprocessing_transform=video_handler._transform
)
scores = video_handler.classify(frame_list=frame_list)
