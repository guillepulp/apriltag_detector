import rclpy
from rclpy.node import Node
from pypylon import pylon
import cv2
from pyapriltags import Detector
import numpy as np
from geometry_msgs.msg import Pose2D


class AprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector')

        self.tag_pose_pub = self.create_publisher(Pose2D, 'tag_pose', 10)

        self.pixel = 0.0214583333
        self.color = (255, 0, 0)
        mtx = [582.112989294283, 0.0, 373.264027997346, 0.0, 582.32443506762, 278.025022753097, 0.0, 0.0, 1.0]
        dist = [-0.301536816015404, 0.0406059177974744, 0.000569907985954076, 0.000168141219425074, 0.134773448317241]

        self.mtx = np.array(mtx).reshape((3, 3))
        self.dist = np.array(dist).reshape((1, 5))

        # conecting to the first available camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Grabing Continusely (video) with minimal delay
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        self.converter = pylon.ImageFormatConverter()

        # converting to opencv bgr format
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)
        self.at_detector = at_detector


    def detect_tags(self):
        while self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                hold = image.GetArray()
                img = cv2.cvtColor(hold, cv2.COLOR_BGR2GRAY)
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h)) 
                dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                tags = self.at_detector.detect(dst, estimate_tag_pose=True, camera_params=(self.mtx[0,0], self.mtx[1,1], self.mtx[0,2], self.mtx[1,2]), tag_size=0.04)
                color_img = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
                p1, p2 = color_img.shape[:2]
                #692 x 488 instead of 720 x 540, 9x7 cm window

                for tag in tags:
                    cosine_for_pitch = np.sqrt(tag.corners[0][0] ** 2 + tag.corners[1][0] ** 2)
                    is_singular = cosine_for_pitch < 10**-6
                    if not is_singular:
                        yaw = np.arctan2(tag.pose_R[1][0], tag.pose_R[0][0])
                    else:
                        yaw = np.arctan2(-tag.pose_R[1][2], tag.pose_R[1][1])

                    print("Tag = ", tag.tag_id, "; Angle = ", -np.rad2deg(yaw), "X, Y: (cm)", np.around(tag.corners[3, 0] * self.pixel, 3), np.around(tag.corners[3, 1] * self.pixel, 3),  "\n")
                    x = np.around(tag.corners[3, 0] * self.pixel, 3)
                    y = np.around(tag.corners[3, 1] * self.pixel, 3)
                    theta = -np.rad2deg(yaw)

                    pose_msg = Pose2D()
                    pose_msg.x = x
                    pose_msg.y = y
                    pose_msg.theta = theta

                    self.tag_pose_pub.publish(pose_msg)

                    for idx in range(len(tag.corners)):
                        cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

                    cv2.putText(color_img, str(np.around(tag.tag_id, 2)),
                                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale=0.8,
                                color=(0, 0, 255))
                    
                    center = tag.center.astype(int)
                    cv2.drawMarker(color_img, tuple(center), self.color, cv2.MARKER_CROSS, 20, 1)
                    cv2.drawMarker(color_img, (346, 244), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
                    cv2.drawMarker(color_img, (tag.corners[3, 0].astype(int), tag.corners[3, 1].astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
                
                cv2.imshow('Detected tags', color_img)
                k = cv2.waitKey(1)
                if k == 27:
                    break        

            grabResult.Release()

    def on_shutdown(self):
        self.camera.StopGrabbing()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetector()
    node.detect_tags()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

