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


import xml.etree.ElementTree as ET
import cv2
import os
import ffmpeg

def getBeeLabels(xml_path):
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

def xml_to_csv(xml_path, video_path):
    # parse XML file and extract label information
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # #if video is .mkv, convert to .mp4
    # convert_to_mp4(video_path)

    # initialize video capture
    cap = cv2.VideoCapture(video_path)

    # iterate through frames of video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bs_x = []
        bs_y = []
        be_x = []
        be_y = []

        # iterate through labels in XML file
        for track in root.findall('track'):
            label = track.attrib['label']
            if label.startswith('bs'):
                # code for finding average position of bs labels
                for point in track.findall('points'):
                  x, y = point.attrib['points'].split(',')
                  bs_x.append(float(x))
                  bs_y.append(float(y))

            elif label.startswith('be'):
                # code for finding average position of be labels
                for point in track.findall('points'):
                  x, y = point.attrib['points'].split(',')
                  be_x.append(float(x))
                  be_y.append(float(y))


        # draw lines on frame based on average positions of labels
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # display frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release video capture and close window
    cap.release()
    cv2.destroyAllWindows()

def convert_to_mp4(mkv_file):
    name, ext = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    ffmpeg.input(mkv_file).output(out_name).run()
    print("Finished converting {}".format(mkv_file))


if __name__ == '__main__':
    xml_to_csv("xml/BeeWaggleVideo_36.xml", "raw_videos/output0036.mkv")
