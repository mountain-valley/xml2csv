import xml.etree.ElementTree as Xet
import cv2
import os
import ffmpeg
import pandas as pd
import numpy as np
import re

FRAME_THRESHOLD = 7  # The number of frames allowed between two starting or two ending labels

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
        point = re.split(';|,', points[0].attrib['points'])
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


def labels_to_waggles_DF(df, n, FPS=30):
    if df.empty:
        print("No labels found in xml file. Skipping this video")
        return df

    labelsStarts = [f'bs{i}' for i in range(20)]
    labelEnds = [f'be{i}' for i in range(20)]
    columns = ['startFrame', 'endFrame', 'angle', 'duration',
               'startPointX', 'startPointY', 'endPointX', 'endPointY',
               'framesStart', 'framesEnd', 'pointsStart', 'pointsEnd']

    df = df.sort_values(by='frame')
    rows = []
    errors = []

    for (beeIdStart, beeIdEnd) in zip(labelsStarts, labelEnds):
        # first get a list of all the instances of labels for the given beeId
        labelList = df[(df['BeeLabel'] == beeIdStart) | (df['BeeLabel'] == beeIdEnd)]
        # labelList is a dataframe with all the label instances of the beeId

        # exit if there are no labels for this beeId
        if labelList.empty:
            print("no labels for beeId {}".format(beeIdStart))
            continue

        # order LabelList by frame
        labelList = labelList.sort_values(
            by='frame')  # this line may be redundant considering the sorting before this for loop

        curBeeLabel = beeIdStart
        lastBeeLabel = None
        lastFrame = int(labelList.iloc[0].frame)
        last_id = int(labelList.iloc[0].Index)

        framesStart = []
        framesEnd = []
        pointsStart = []
        pointsEnd = []

        last_label_index = int(labelList.iat[-1, 1])

        # calculate angle and duration of waggles.
        for i, row in labelList.iterrows():
            curBeeLabel = row['BeeLabel']
            frame = int(row.frame)
            point = np.array(row.Point[0:2]).astype(float)
            id = int(row.Index)

            # if current label is a start label and the last label was an end label, then record the last waggle and
            # reset the lists to start a new waggle
            if (curBeeLabel == beeIdStart and lastBeeLabel == beeIdEnd) or (
                    i == last_label_index and curBeeLabel == beeIdEnd):

                if i == last_label_index:
                    framesEnd.append(frame)
                    pointsEnd.append(point)

                # record the waggle
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
                # make sure the current frame is not too far from the last frame
                if frame - lastFrame > FRAME_THRESHOLD:
                    errors.append([curBeeLabel, lastFrame, frame, 'start'])
                framesStart.append(frame)
                pointsStart.append(point)

            # on end points
            elif curBeeLabel == beeIdEnd:
                if frame - lastFrame > FRAME_THRESHOLD:
                    errors.append([curBeeLabel, lastFrame, frame, 'end'])
                if len(framesStart) == 0:
                    print(f"Warning: end label without start label. Skipping end label. Occurred at frame {frame} with label {beeIdEnd}")
                    continue
                else:
                    framesEnd.append(frame)
                    pointsEnd.append(point)

            lastBeeLabel = curBeeLabel
            lastFrame = frame
            last_id = id

    df = pd.DataFrame(rows, columns=columns)
    columns = ['startFrame', 'endFrame', 'angle', 'duration',
               'startPointX', 'startPointY', 'endPointX', 'endPointY', ]
    df = df.sort_values(by='startFrame')
    df[columns].to_csv(f'csv/WaggleDance_{n}_Labels.csv')

    errors.sort(key=lambda i: i[1])
    for line in errors:
        print(f"for label \"{line[0]}\", \"{line[1]}\" / \"{line[2]}\" {line[3]} frames too far from each other")

    return df[columns]


def xml_to_df(xml_path, n):
    label_df = getBeeLabels(xml_path)
    waggles_df = labels_to_waggles_DF(label_df, n)
    return waggles_df


def df_to_mp4(df, video_path, n):
    if df.empty:
        return
    # Read the .mkv file
    cap = cv2.VideoCapture(video_path)

    # Get the frames properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        print("Error reading video file. Skipping this video")
        return
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # `height`

    # create the video writer
    print(f'fps: {fps}, frames_count: {frames_count}, width: {width}, height: {height}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'annotated/labelsVideo_{n}.mp4', fourcc, fps, (width, height))

    curr_frame_num = 0  # Keeps track of the last frame loaded. The xml file indexes from 0
    begin_dict = 0
    end_dict = 0
    # add 0th frame to the dictionary
    ret, frame = cap.read()
    if not ret:
        print("Error reading frames. Skipping this video")
        return
    frames = {0: frame}

    print('looping through waggle data and drawing lines')
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


def xml_to_mp4(xml_path, video_path, n, long_lines=False):
    df = xml_to_df(xml_path, n)
    out = df_to_mp4(df, video_path, n)
    return out


def convert_to_mp4(mkv_file, n):
    name, ext = os.path.splitext(mkv_file)
    out_name = "labelsVideo_" + n + ".mp4"
    ffmpeg.input(mkv_file).output(out_name).run()
    print("Finished converting {}".format(mkv_file))


if __name__ == '__main__':
    # n = ['01']
    # n = ['00b', '01', '01b', '02', '02b', '10', '12', '40', '41']
    # for i in n:
    #     xml_to_mp4(f'xml/BeeWaggleVideo_{i}.xml', f'raw_videos/output00{i}.mkv', i, long_lines=True)
    #     print(f"finished video {i}")
    #
    # revised = [11, 36, 38, 39, 45]
    # for i in revised:
    #     xml_to_mp4(f'xml/BeeWaggleVideo_{i}_revised.xml', f'raw_videos/output00{i}.mkv', i, long_lines=True)
    #     print(f"finished video {i}")

    revised = [36]
    for i in revised:
        xml_to_mp4(f'xml/BeeWaggleVideo_{i}_revised.xml', f'raw_videos/output00{i}.mkv', i, long_lines=True)
        print(f"finished video {i}")