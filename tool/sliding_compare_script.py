import cv2
import numpy as np
import os.path as osp
import glob

img_height, img_width = 384, 384 # 256, 256
padding = 64
video_height, video_width = 512, 512


def combTwoSeqs(input_img_path, stop_frame, seq1_dir, seq2_dir, file_pattern1, file_pattern2, compare_method, ours_method, video_name='video.avi'):
    input_img = cv2.imread(input_img_path)
    input_img = cv2.resize(input_img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

    # images to be converted to a video
    seq1_images = sorted(glob.glob(seq1_dir + '/' + file_pattern1))
    seq2_images = sorted(glob.glob(seq2_dir + '/' + file_pattern1))
    # depths to be converted to a video
    seq1_depths = sorted(glob.glob(seq1_dir + '/' + file_pattern2))
    seq2_depths = sorted(glob.glob(seq2_dir + '/' + file_pattern2))

    template_image = np.zeros((video_height, video_width * 3, 3), dtype=np.uint8)
    

    # assume 15 fps video
    # name for the saving video
    # video_name = osp.join(file_dir, video_name)
    video = cv2.VideoWriter(video_name, 0, 20, (video_width*3, video_height))

    # write seq1 images into the video stream
    for seq1_image_path, seq1_depth_path in zip(seq1_images, seq1_depths):
        tmp = template_image.copy()
        tmp[padding:512-padding, 512*0+padding:512*1-padding] = input_img
        # print(int(video_width * 1 // 2 - 60))
        cv2.putText(tmp, 'input image', (int(video_width * 1 // 2 - 100), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

        img = cv2.imread(seq1_image_path)
        resized_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        tmp[padding:512-padding, 512*1+padding:512*2-padding] = resized_img
        depth = cv2.imread(seq1_depth_path)
        resized_depth = cv2.resize(depth, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        tmp[padding:512-padding, 512*2+padding:512*3-padding] = resized_depth

        cv2.putText(tmp, 'RGB', (int(video_width * 1 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(tmp, 'Depth', (int(video_width * 2 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(tmp, compare_method, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

        video.write(tmp)

    # half half
    for idx, (seq1_image_path, seq1_depth_path, seq2_image_path, seq2_depth_path) in enumerate(zip(seq1_images, seq1_depths, seq2_images, seq2_depths)):
        tmp1 = template_image.copy()
        tmp1[padding:512-padding, 512*0+padding:512*1-padding] = input_img
        img = cv2.imread(seq1_image_path)
        resized_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        tmp1[padding:512-padding, 512*1+padding:512*2-padding] = resized_img
        depth = cv2.imread(seq1_depth_path)
        resized_depth = cv2.resize(depth, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        tmp1[padding:512-padding, 512*2+padding:512*3-padding] = resized_depth

        # cv2.putText(tmp1, compare_method, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

        tmp2 = template_image.copy()
        tmp2[padding:512-padding, 512*0+padding:512*1-padding] = input_img
        img = cv2.imread(seq2_image_path)
        resized_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        tmp2[padding:512-padding, 512*1+padding:512*2-padding] = resized_img
        depth = cv2.imread(seq2_depth_path)
        resized_depth = cv2.resize(depth, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        tmp2[padding:512-padding, 512*2+padding:512*3-padding] = resized_depth

        # cv2.putText(tmp2, ours_method, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

        cv2.putText(tmp1, compare_method, (512*2-180, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
        cv2.putText(tmp1, compare_method, (512*3-180, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
        cv2.putText(tmp2, ours_method, (512*1+40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))
        cv2.putText(tmp2, ours_method, (512*2+40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255))

        line_width = 2
        line1_idx = video_width / len(seq1_images) * (idx+1)
        line1_idx = 512*1 + int(line1_idx) - line_width

        line2_idx = video_width / len(seq1_depths) * (idx+1)
        line2_idx = 512*2 + int(line2_idx) - line_width

        tmp = template_image.copy()
        tmp[:, line1_idx+line_width:] = tmp1[:, line1_idx+line_width:]
        tmp[:, line1_idx:line1_idx+line_width] = np.array([0, 0, 255], dtype=np.uint8)
        tmp[:, :line1_idx] = tmp2[:, :line1_idx]

        tmp[:, line2_idx+line_width:] = tmp1[:, line2_idx+line_width:]
        tmp[:, line2_idx:line2_idx+line_width] = np.array([0, 0, 255], dtype=np.uint8)
        tmp[:, 1024:line2_idx] = tmp2[:, 1024:line2_idx]

        cv2.putText(tmp, 'input image', (int(video_width * 1 // 2 - 100), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(tmp, 'RGB', (int(video_width * 1 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(tmp, 'Depth', (int(video_width * 2 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        
        video.write(tmp)

        # Special
        if idx == stop_frame:
            save_line1_idx, save_line2_idx = line1_idx, line2_idx

            while line1_idx > (512+512//4):
                line1_idx = line1_idx - line_width*4
                line2_idx = line2_idx - line_width*4

                tmp = template_image.copy()
                tmp[:, line1_idx+line_width:] = tmp1[:, line1_idx+line_width:]
                tmp[:, line1_idx:line1_idx +
                    line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, :line1_idx] = tmp2[:, :line1_idx]

                tmp[:, line2_idx+line_width:] = tmp1[:, line2_idx+line_width:]
                tmp[:, line2_idx:line2_idx + line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, 1024:line2_idx] = tmp2[:, 1024:line2_idx]

                cv2.putText(tmp, 'input image', (int(video_width * 1 // 2 - 100), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'RGB', (int(video_width * 1 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'Depth', (int(video_width * 2 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        
                video.write(tmp)

            while line1_idx < (512+512//4*3):
                line1_idx = line1_idx + line_width*4
                line2_idx = line2_idx + line_width*4

                tmp = template_image.copy()
                tmp[:, line1_idx+line_width:] = tmp1[:, line1_idx+line_width:]
                tmp[:, line1_idx:line1_idx +
                    line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, :line1_idx] = tmp2[:, :line1_idx]

                tmp[:, line2_idx+line_width:] = tmp1[:, line2_idx+line_width:]
                tmp[:, line2_idx:line2_idx +
                    line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, 1024:line2_idx] = tmp2[:, 1024:line2_idx]

                cv2.putText(tmp, 'input image', (int(video_width * 1 // 2 - 100), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'RGB', (int(video_width * 1 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'Depth', (int(video_width * 2 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        
                video.write(tmp)

            while line1_idx > (512+512//4):#save_line1_idx:
                line1_idx = line1_idx - line_width*4
                line2_idx = line2_idx - line_width*4

                tmp = template_image.copy()
                tmp[:, line1_idx+line_width:] = tmp1[:, line1_idx+line_width:]
                tmp[:, line1_idx:line1_idx +
                    line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, :line1_idx] = tmp2[:, :line1_idx]

                tmp[:, line2_idx+line_width:] = tmp1[:, line2_idx+line_width:]
                tmp[:, line2_idx:line2_idx +
                    line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, 1024:line2_idx] = tmp2[:, 1024:line2_idx]

                cv2.putText(tmp, 'input image', (int(video_width * 1 // 2 - 100), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'RGB', (int(video_width * 1 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'Depth', (int(video_width * 2 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        
                video.write(tmp)

            while line1_idx < save_line1_idx:
                line1_idx = line1_idx + line_width*4
                line2_idx = line2_idx + line_width*4

                tmp = template_image.copy()
                tmp[:, line1_idx+line_width:] = tmp1[:, line1_idx+line_width:]
                tmp[:, line1_idx:line1_idx +
                    line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, :line1_idx] = tmp2[:, :line1_idx]

                tmp[:, line2_idx+line_width:] = tmp1[:, line2_idx+line_width:]
                tmp[:, line2_idx:line2_idx +
                    line_width] = np.array([0, 0, 255], dtype=np.uint8)
                tmp[:, 1024:line2_idx] = tmp2[:, 1024:line2_idx]

                cv2.putText(tmp, 'input image', (int(video_width * 1 // 2 - 100), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'RGB', (int(video_width * 1 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                cv2.putText(tmp, 'Depth', (int(video_width * 2 + video_width // 2 - 40), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        
                video.write(tmp)

    # write seq1 images into the video stream
    for seq2_image_path, seq2_depth_path in zip(seq2_images, seq2_depths):
        tmp = template_image.copy()
        tmp[padding:512-padding, 512*0+padding:512*1-padding] = input_img
        img = cv2.imread(seq2_image_path)
        resized_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        tmp[padding:512-padding, 512*1+padding:512*2-padding] = resized_img
        depth = cv2.imread(seq2_depth_path)
        resized_depth = cv2.resize(depth, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        tmp[padding:512-padding, 512*2+padding:512*3-padding] = resized_depth

        cv2.putText(tmp, ours_method, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(tmp, 'input image', (int(video_width * 1 // 2 - 100), video_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        video.write(tmp)

    cv2.destroyAllWindows()
    video.release()

    print('Video saved in ', video_name)


if __name__ == '__main__':
    
    object_name = 'DexYCB_Drill' #'HO3D_Scissors'

    compare_method = 'PixelNeRF'
    
    compare_dir = '../output/exp_02-28_22:00/vis/20200709_151632_000030_rotating_epoch200'#../output/exp_03-05_23:54/vis/GSF1_0520_rotating_epoch299'

    ours_method = 'HandNeRF (Ours)'
    ours_dir = '../output/exp_03-04_13:58/vis/20200709_151632_000030_rotating_epoch99' #'../output/exp_03-05_23:53/vis/GSF1_0520_rotating_epoch299'

    # input_img_path = osp.join('../data/HanCo/data', 'rgb/0693/cam0/00000024.jpg')
    input_img_path = osp.join('../data/DexYCB/data', '20200820-subject-03/20200820_135508/836212060125/color_000060.jpg')
    # input_img_path = osp.join('../data/DexYCB/data', '20200709-subject-01/20200709_151632/840412060917/color_000030.jpg')
    # input_img_path = osp.join('../data/HO3D/data', 'train/GSF10/rgb/0520.jpg')


    file_pattern1 = 'Rotating_RGB_0*.jpg'
    file_pattern2 = 'Rotating_Depth_0*.jpg'
    
    stop_frame = 3

    video_name = f'{compare_method}_{ours_method}_{object_name}_rotate.avi'

    combTwoSeqs(input_img_path, stop_frame, compare_dir, ours_dir, file_pattern1, file_pattern2, compare_method, ours_method, video_name)