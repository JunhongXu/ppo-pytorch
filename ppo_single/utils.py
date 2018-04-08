import cv2
import imageio


def convert_gif(video_name, to_path):
    # read video into images
    video = cv2.VideoCapture(video_name)
    frame_cnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
    with imageio.get_writer(to_path, duration=1/24) as writer:
        index = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128))
            index += 1
            writer.append_data(frame)
            print('\rProgress: %.4f' % (index / frame_cnt), flush=True, end='')


if __name__ == '__main__':
    convert_gif('/media/junhong/Data/torcsVideo/out.ogv', 'imgs/torcs.gif')
