# I need to write a script in python that takes a .mkv video and an xml file and combines them. The .xml file contains
# label information for the video. The video contains bee waggle dances. The xml file contains labels. Each label is in
# following format:
#    <track id="11" label="bs5" source="manual">
#     <points frame="117" outside="0" occluded="0" keyframe="1" points="801.62,568.35" z_order="0">
#     </points>
#     <points frame="118" outside="1" occluded="0" keyframe="1" points="801.62,568.35" z_order="0">
#     </points>
#   </track>
# The output will be a .mkv file with the labels represented as lines with a dot in the middle. Each line can be formed
# using two or more labels in the xml file. The bs# label(s) represent the beginning of a line and be# labels represent
# the end of a line. The label numbers match up with each other. For example, bs5 and be5 are the beginning and end of a
# a line. The output video will have a line drawn between the bs5 and be5 labels. However, there may be multiple #bs5
# labels and multiple #be5 labels. If multiple #bs5 labels exist, then the average position of all the #bs5 labels will
# be used. If multiple #be5 labels exist, then the average position of all the #be5 labels will be used. Labels are
# reused so labels are only averaged if they are found within a few frames of eachother.


import xml.etree.ElementTree as Xet
import cv2
import os
import ffmpeg
import pandas as pd
import numpy as np
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from icecream import ic
from collections import deque


def getBeeLabels(xml_path):
    """
    This function takes an xml file and returns a pandas dataframe with the following columns:
    frame, label, x, y
    :param xml_path: path to xml file
    :return: pandas dataframe
    """
    rows = []
    cols = ['BeeLabel', 'Index', 'frame', 'Point']
    # Parsing the XML file
    xmlparse = Xet.parse(xml_path)
    root = xmlparse.getroot()
    for track in xmlparse.findall('track'):
        beeLabel = track.attrib['label']
        index = track.attrib['id']
        points = track.findall('points')
        if len(points) == 0:
            continue
        frame = int(points[0].attrib['frame'])
        point = points[0].attrib['points'].split(',')
        newRow = [beeLabel, index, frame, point]
        rows.append(newRow)
    df = pd.DataFrame(rows, columns=cols)
    return df


def getAngle(p1, p2):
    v = np.array(p2) - np.array(p1)
    return np.arctan(v[1] / v[0])  # TODO: account for dividing by zero


def createWagglesDF(df, FPS=30, waggle=41):
    labelsStarts = [f'bs{i}' for i in range(10)]
    labelEnds = [f'be{i}' for i in range(10)]
    columns = ['startFrame', 'endFrame', 'angle', 'duration',
               'startPointX', 'startPointY', 'endPointX', 'endPointY',
               'framesStart', 'framesEnd', 'pointsStart', 'pointsEnd']

    rows = []
    for beeId in zip(labelsStarts, labelEnds):
        # first get a list of all the instances of labels for the given beeId
        # beeId is a tuple in the format of (bs#, be#)
        beeIdStart, beeIdEnd = beeId
        beeDfStart, beeDfEnd = df[df['BeeLabel'] == beeIdStart].copy(), df[df['BeeLabel'] == beeIdEnd].copy()
        labelList = pd.concat([beeDfEnd, beeDfStart])
        labelList.sort_values(by='frame', ignore_index=True, inplace=True)  # labelList is a dataframe with all the
        # label instances of the beeId, ordered by frame

        curBeeLabel = beeIdStart
        lastBeeLabel = beeIdEnd

        framesStart = []
        framesEnd = []
        pointsStart = []
        pointsEnd = []

        # calulate angle and duration of waggles.
        for i, row in labelList.iterrows():
            curBeeLabel = row['BeeLabel']
            frame = int(row.frame)
            point = np.array(row.Point).astype(float)

            # if the current label is a start label and the last label was an end label, then we are starting a new
            # waggle
            if (curBeeLabel == beeIdStart and lastBeeLabel == beeIdEnd) or i - 1 == len(labelList):
                # create new row
                if len(framesEnd) > 0:  # ensure that there are end frames
                    # average the start and end frames and the start and end points of the last waggle
                    startFrame = int(np.mean(framesStart))
                    endFrame = int(np.mean(framesEnd))
                    startPoint = np.mean(np.array(pointsStart), axis=0)
                    endPoint = np.mean(np.array(pointsEnd), axis=0)

                    duration = (endFrame - startFrame) / FPS
                    angle = getAngle(startPoint, endPoint)

                    newRow = [startFrame, endFrame, angle, duration, startPoint[0], startPoint[1], endPoint[0],
                              endPoint[1], framesStart, framesEnd, pointsStart, pointsEnd]
                    rows.append(newRow)

                # reset lists with info from new waggle
                framesStart = [frame]
                pointsStart = [point]
                framesEnd = []
                pointsEnd = []

            # Switching to end points (may be redundant/unnecessary)
            elif lastBeeLabel == beeIdStart and curBeeLabel == beeIdEnd:
                framesEnd = [frame]
                pointsEnd = [point]

            # On start points
            elif curBeeLabel == beeIdStart:
                framesStart.append(frame)
                pointsStart.append(point)

            # on end points
            elif curBeeLabel == beeIdEnd:
                framesEnd.append(frame)
                pointsEnd.append(point)

            lastBeeLabel = curBeeLabel

    df = pd.DataFrame(rows, columns=columns)
    columns = ['startFrame', 'endFrame', 'angle', 'duration',
               'startPointX', 'startPointY', 'endPointX', 'endPointY', ]
    df = df.sort_values(by='startFrame')
    df[columns].to_csv(f'WaggleDance_{waggle}_Labels.csv')
    return df[columns]


def xml_to_df(xml_path, video_path):
    label_df = getBeeLabels(xml_path)
    waggles_df = createWagglesDF(label_df)
    return waggles_df


def df_to_mp4(df, video_path):
    # Read the .mkv file
    cap = cv2.VideoCapture(video_path)

    # Get the frames properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # `height`

    # create the video writer
    print(f'fps: {fps}, frames_count: {frames_count}, width: {width}, height: {height}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    curr_frame_num = 0  # the xml file indexes from 0
    begin_dict = 0
    end_dict = 0
    frames = {}

    print('looping through dataframe')
    # Loop through the DataFrame
    for i, row in df.iterrows():
        start_frame = int(row['startFrame'])
        end_frame = int(row['endFrame'])
        angle = row['angle']
        start_point = (int(row['startPointX']), int(row['startPointY']))
        end_point = (int(row['endPointX']), int(row['endPointY']))

        # Read the frames until you reach the frame in question
        while curr_frame_num < start_frame:
            ret, frame = cap.read()
            if not ret:
                break
            curr_frame_num += 1
            out.write(frame)

        # write the frames that will no longer be used (the dataframe is sorted by startFrame so if the startFrame is
        # greater than the beginning of the dictionary, then we no longer will need those frames)
        while begin_dict < start_frame and len(frames) > 0:
            out.write(frames.pop(begin_dict))
            begin_dict += 1

        # Draw the lines on the relevant frames
        for i in range(start_frame, end_frame):
            # if the frame is already in the dictionary, then get it from the dict, otherwise read in the frame and add
            # it to the dict
            if i <= end_dict:
                frame = frames[i]
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                curr_frame_num += 1
                end_dict = i
                frames[i] = frame

            # Draw the line
            # x = int(start_point[0] + (i - start_frame) * (end_point[0] - start_point[0]) / (end_frame - start_frame))
            # y = int(angle * x + y_intercept)
            # cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), 5)
            cv2.line(frame, start_point, end_point, (255, 0, 0), 5)
            cv2.putText(
                img=frame,
                text='Frame: {}'.format(curr_frame_num),
                org=(200, 200),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3)
            out.write(frame)

    print('done drawing lines')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return out


def xml_to_mp4(xml_path, video_path):
    df = xml_to_df(xml_path, video_path)
    out = df_to_mp4(df, video_path)
    return out


def convert_to_mp4(mkv_file):
    name, ext = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    ffmpeg.input(mkv_file).output(out_name).run()
    print("Finished converting {}".format(mkv_file))


if __name__ == '__main__':
    xml_to_mp4("xml/BeeWaggleVideo_39_revised_2.xml", "raw_videos/output0039.mkv")
