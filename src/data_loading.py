"""
@File   :   data_loading.py
@Date   :   2023/7/14
@Description    :   This file is for slicing the time series data into windows
"""

from typing import Optional, TextIO
import re
import numpy as np


class Slider:
    """
    A stack for sliding the window
    """
    def __init__(self, window_size=40, step_size=20):
        self.tmp = []
        self.WINDOW_SIZE = window_size
        self.STEP_SIZE = step_size
        self.activity_code = ' '

    def push(self, data, input_activity_code) -> Optional[list[list]]:
        if input_activity_code != self.activity_code:
            self.tmp = []
            self.tmp.append(data)
            self.activity_code = input_activity_code
            return None
        else:
            self.tmp.append(data)
            return self.update()

    def update(self) -> Optional[list[list]]:
        if len(self.tmp) == self.WINDOW_SIZE:
            # send to stage of feature extraction
            window = self.tmp
            self.tmp = self.tmp[self.STEP_SIZE: self.WINDOW_SIZE]
            return window

        else:
            return None


def write2log(subject_id, activity_code, content, output: TextIO):
    output.write(f"subject_id-{subject_id}, activity_code-{activity_code}: ")
    output.write(str(content))
    output.write("\n\n")


def label_parser(alpha):
    """
    This is an auxiliary function to fix the lost of activity code "N" in the WIDSM dataset
    @param alpha: activity code
    @return:
    """
    alpha = ord(alpha)
    if alpha > ord('N'):
        alpha -= 1

    return alpha - ord('A')


def txt2sliced(txt_filepath='data/WISDM-Dataset-Watch-Gyro-Raw/data_test_gyro_watch.txt',
               sliced_filepath='temp/sliced-data/test.txt'):
    data = []
    label = []
    slider = Slider()

    with open(sliced_filepath, 'w') as sliced:
        with open(txt_filepath, 'r') as txt:
            for line in txt.readlines():
                # filter by regularization
                d = re.split(r"[,;]", line)

                # create sliding windows (3-axis)
                window = slider.push(data=d[3:3 + 3],
                                     input_activity_code=d[1])

                # generate features
                if window is not None:
                    sensor_data = np.array(window).transpose().astype(float)
                    # append
                    data.append(sensor_data)
                    label.append(label_parser(d[1]))

                    # store windows
                    write2log(subject_id=d[0], activity_code=d[1],
                              content=sensor_data, output=sliced)
    return data, label
