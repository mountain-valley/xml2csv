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


# labels = [f'b{i}' for i in range(20)]
# rows = []
#
# labelsStarts = [f'bs{i}' for i in range(20)]
# labelEnds = [f'be{i}' for i in range(20)]
# labels = labelsStarts + labelEnds

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
    return np.arctan(v[1] / v[0])


# def createVideoPackage(df, FPS=30, waggle=41):
#     columns = ['startFrame', 'endFrame', 'angle', 'duration',
#                'startPointX', 'startPointY', 'endPointX', 'endPointY']
#     # os.makeDir(dirName)
#     beeNum = 0
#     rows = []
#     for beeId in labels:
#         beeDf = df[df['BeeLabel'] == beeId].copy()
#         beeDf.sort_values(by='frame', ignore_index=True, inplace=True)
#         print(beeDf)
#         for i, row in beeDf.iterrows():
#             if i % 2 == 0:
#                 startFrame = int(row.frame)
#                 startPoint = np.array(row.Point).astype(float)
#             else:
#                 endFrame = int(row.frame)
#                 endPoint = np.array(row.Point).astype(float)
#                 duration = (endFrame - startFrame) / FPS
#                 angle = getAngle(startPoint, endPoint)
#                 newRow = [startFrame, endFrame, angle, duration, startPoint[0], startPoint[1], endPoint[0], endPoint[1]]
#                 rows.append(newRow)
#     df = pd.DataFrame(rows, columns=columns)
#     df.to_csv(f'WaggleDance_{waggle}_Labels.csv')


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
    df[columns].to_csv(f'WaggleDance_{waggle}_Labels.csv')
    return df[columns]


# def createVideoPackageWithRuns(df, dirName, FPS=30):
#     columns = ['StartFrame', 'EndFrame', 'ManualRunAngle_Deg', 'RunTime_Sec']
#     os.makeDir(dirName)
#     for beeId in labels:
#         beeDf = df[df['BeeLabel'] == beeId].copy()
#         rows = []
#         for i, row in beeDf.iterrows():
#             if i % 2 == 0:
#                 startFrame = row.frame
#                 startPoint = row.point
#             else:
#                 endFrame = row.frame
#                 endPoint = row.point
#                 duration = (endFrame - startFrame) / FPS
#                 angle = getAngle(startPoint, endPoint)
#                 newRow = [startFrame, endFrame, angle, duration]
#                 rows.append(newRow)
#             df = pd.DataFrame(rows, columns=columns)


def xml_to_df(xml_path, video_path):
    label_df = getBeeLabels(xml_path)
    waggles_df = createWagglesDF(label_df)
    return waggles_df

def df_to_mp4(df, video_path):
    # Read the .mkv file
    cap = cv2.VideoCapture(video_path)

    # Get the frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Loop through the DataFrame
    for i, row in df.iterrows():
        start_frame = row['startFrame']
        end_frame = row['endFrame']
        angle = row['angle']
        start_point = (row['startPointX'], row['startPointY'])
        end_point = (row['endPointX'], row['endPointY'])

        # Select the frames
        for j in range(start_frame, end_frame):
            # Draw the line
            cv2.line(frames[j], start_point, end_point, (255, 0, 0), 2)
            # Put the frame number on the frame
            cv2.putText(frames[j], 'Frame: {}'.format(j), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()
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
    xml_to_mp4("xml/BeeWaggleVideo_36.xml", "raw_videos/output0036.mkv")
