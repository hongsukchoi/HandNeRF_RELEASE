import cv2
import os.path as osp
import glob


def images2video(file_dir='../output/exp_01-05_18:06/vis', file_pattern='InputIs*.jpg', video_name='video.avi'):
    # images to be converted to a video
    images = sorted(glob.glob(file_dir + '/' + file_pattern))
    # name for the saving video
    video_name = osp.join(file_dir, video_name)

    # assume 30 fps video
    # frame = cv2.imread(osp.join(images[0]))
    # height, width, layers = frame.shape
    height, width = 384, 384
    video = cv2.VideoWriter(video_name, 0, 15, (width, height))

    # write images into the video stream
    for image in images:
        img = cv2.imread(image)
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        video.write(resized_img)
    cv2.destroyAllWindows()
    video.release()

    print('Video saved in ', video_name)


if __name__ == '__main__':
    file_dir = '../output/exp_02-27_23:57/vis/20200820_135508_000060_rotating_epoch299'
    file_pattern = 'Rotating_*.jpg'
    video_name = 'RGB_DexYCB_Can_HandNeRF_rotate.avi'

    images2video(file_dir, file_pattern, video_name)