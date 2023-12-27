import os
import re
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def processVideo(videoPath, videoName, savePath, scaleFactor):
    # Load video using OpenCV
    cap = cv2.VideoCapture(videoPath + "/" + videoName)

    # Create Matplotlib figure
    fig, ax = plt.subplots()
    curr = 1
    save_directory = savePath
    os.makedirs(save_directory, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame based on the scale factor
        new_width = int(frame.shape[1] * scaleFactor)
        new_height = int(frame.shape[0] * scaleFactor)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Display frame using Matplotlib
        ax.imshow(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {curr}')  # Set frame number as title

        # plt.draw()
        # plt.pause(0.001)

        frame_path = os.path.join(save_directory, f'frame_{curr}.png')
        cv2.imwrite(frame_path, resized_frame)
        # Clear the previous frame
        ax.cla()
        curr += 1

    # Release the video capture and close the Matplotlib window
    cap.release()
    plt.close()



def imageToFiles(image, black, white):
    counter = 1
    directory = 'theFunny'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for row in image:
        for pixel in row:
            try:
                if pixel == 0:
                    black.save(f"{directory}/{counter}.png")
                else:
                    white.save(f"{directory}/{counter}.png")
            except Exception as e:  # To handel the exceptions if they happen
                print(f"Error encountered: {e}")
            counter += 1


def readGrayScaleImages(path):
    # Example usage
    images = sorted(os.listdir(path), key=lambda x: int(re.findall(r'\d+', x)[0]))
    black_image = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
    white_image = Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * 255)
    for image_file in images:
        image_path = os.path.join(path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (24, 16))  # Resize to 24x16 since that is what my screen size allows
        imageToFiles(resized_image, black_image, white_image)
        print(f"Finished {image_file}")
    plt.close()


if __name__ == '__main__':
    imagePath = 'images'
    videoPath = 'video'
    videoName = 'badApple.mp4'
    scale_factor = 0.02
    # processVideo(videoPath, videoName, imagePath, scale_factor)
    readGrayScaleImages(imagePath)

    print("Completed")
