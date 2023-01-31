import xml.etree.ElementTree as Xet
import cv2
import os
import ffmpeg
import pandas as pd
import numpy as np


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
    # account for dividing by zero
    if v[0] == 0:
        v[0] = 0.0001
    return np.arctan(v[1] / v[0])


def createWagglesDF(df, FPS=30, waggle=41):
    labelsStarts = [f'bs{i}' for i in range(10)]
    labelEnds = [f'be{i}' for i in range(10)]
    columns = ['startFrame', 'endFrame', 'angle', 'duration',
               'startPointX', 'startPointY', 'endPointX', 'endPointY',
               'framesStart', 'framesEnd', 'pointsStart', 'pointsEnd']

    df = df.sort_values(by='frame')
    rows = []
    for (beeIdStart, beeIdEnd) in zip(labelsStarts, labelEnds):
        # first get a list of all the instances of labels for the given beeId
        labelList = df[(df['BeeLabel'] == beeIdStart) | (df['BeeLabel'] == beeIdEnd)]
        # labelList is a dataframe with all the label instances of the beeId
        # order LabelList by frame (
        labelList = labelList.sort_values(by='frame')  # this line may be redundant considering the sorting before this for loop

        curBeeLabel = beeIdStart
        lastBeeLabel = beeIdEnd

        framesStart = []
        framesEnd = []
        pointsStart = []
        pointsEnd = []

        last_label_index = int(labelList.iat[-1, 1])

        # calculate angle and duration of waggles.
        for i, row in labelList.iterrows():
            curBeeLabel = row['BeeLabel']
            frame = int(row.frame)
            point = np.array(row.Point).astype(float)

            # if the current label is a start label and the last label was an end label, then we are starting a new
            # waggle
            if (curBeeLabel == beeIdStart and lastBeeLabel == beeIdEnd) or (
                    i == last_label_index and curBeeLabel == beeIdEnd):
                # create new row
                if i == last_label_index:
                    framesEnd.append(frame)
                    pointsEnd.append(point)
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

            # Switching to end points (maybe redundant/unnecessary)
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
    df[columns].to_csv(f'WaggleDance_Labels.csv')
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

    curr_frame_num = 0  # Keeps track of the last frame loaded. The xml file indexes from 0
    begin_dict = 0
    end_dict = 0
    # add 0th frame to the dictionary
    ret, frame = cap.read()
    frames = {0: frame}

    print('looping through waggle data and drawing lines'
          '')
    # Loop through the DataFrame
    for i, row in df.iterrows():
        start_frame = int(row['startFrame'])
        end_frame = int(row['endFrame'])
        angle = row['angle']
        start_point = (int(row['startPointX']), int(row['startPointY']))
        end_point = (int(row['endPointX']), int(row['endPointY']))

        # extend the dictionary to include the start and end frames
        while end_dict < end_frame:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            end_dict += 1
            frames[end_dict] = frame
            # write the frame number
            cv2.putText(
                img=frame,
                text='Frame: {}'.format(end_dict),
                org=(200, 200),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3)

        # shorten the dictionary to include only the start and end frames
        while begin_dict < start_frame:
            out.write(frames.pop(begin_dict))
            begin_dict += 1

        # Draw the lines on the relevant frames
        for i in range(start_frame, end_frame + 1):
            frame = frames[i]
            # Draw the line
            # x = int(start_point[0] + (i - start_frame) * (end_point[0] - start_point[0]) / (end_frame - start_frame))
            # y = int(angle * x + y_intercept)
            # cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), 5)
            cv2.line(frame, start_point, end_point, (255, 0, 0), 5)

    print('done drawing lines')
    print('writing remaining frames')

    # Write the remaining frames from the dictionary
    while begin_dict <= end_dict:
        out.write(frames.pop(begin_dict))
        begin_dict += 1

    # read and write all the remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    print('done writing all frames')

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return out


def xml_to_mp4(xml_path, video_path, long_lines=False):
    df = xml_to_df(xml_path, video_path)
    out = df_to_mp4(df, video_path)
    return out


def convert_to_mp4(mkv_file):
    name, ext = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    ffmpeg.input(mkv_file).output(out_name).run()
    print("Finished converting {}".format(mkv_file))


if __name__ == '__main__':
    xml_to_mp4("xml/BeeWaggleAnnotations_38_revised.xml", "raw_videos/output0038.mkv", long_lines=True)
