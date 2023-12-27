import os
import re
import threading
import time
import cv2
import matplotlib.pyplot as plt
import sys
import subprocess
from moviepy.editor import VideoFileClip


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


def setCommandPromptSize(width, height):
    # Windows command to set command prompt size
    command = f"mode con: cols={width} lines={height}"

    # Execute the command
    subprocess.call(command, shell=True)


def imageToCommandLine(image):
    for row in image:
        line = ''.join(['*' if pixel == 0 else '.' for pixel in row])
        sys.stdout.write(line + '\n')
    sys.stdout.flush()


def playAudioFromMP4(filepath):
    # Load the video clip
    clip = VideoFileClip(filepath)

    # Extract the audio
    audio = clip.audio

    # Play the audio
    audio.preview()


def readGrayScaleImages(path):
    # Example usage
    width = 50  # Set the desired width in characters
    height = 29  # Set the desired height in lines
    setCommandPromptSize(width, height)
    images = sorted(os.listdir(path), key=lambda x: int(re.findall(r'\d+', x)[0]))
    delay = 1 / 30
    for image_file in images:
        image_path = os.path.join(path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # os.system("cls")
        imageToCommandLine(image)
        time.sleep(delay)

    plt.close()


if __name__ == '__main__':
    imagePath = 'images'
    videoPath = 'video'
    videoName = 'badApple.mp4'
    scale_factor = 0.02
    # processVideo(videoPath, videoName, imagePath, scale_factor)

    thread = threading.Thread(target=playAudioFromMP4, args=(videoPath + "/" + videoName,))
    thread.daemon = True
    # Start the thread
    thread.start()
    readGrayScaleImages(imagePath)

    print("Completed")
    #print("")
