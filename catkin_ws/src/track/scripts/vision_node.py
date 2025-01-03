import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        rospy.logdebug("Image received: %dx%d", cv_image.shape[1], cv_image.shape[0])
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s", e)
    except Exception as e:
        rospy.logerr("Error processing image: %s", e)

def main():
    rospy.init_node('vision_node', log_level=rospy.INFO)
    try:
        rospy.Subscriber("camera/image_raw", Image, image_callback, queue_size=1)
        rospy.loginfo("Vision node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Vision node stopped")

if __name__ == '__main__':
    main()