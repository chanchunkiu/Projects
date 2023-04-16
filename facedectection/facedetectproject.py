from gpiozero import MotionSensor
import cv2
pir = MotionSensor(4)
cam = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

while True:
    print("please wave your hand on the sensor")
    pir.wait_for_motion()
    print("please wait for 2 seconds")
    pir.wait_for_no_motion()
    ret,image = cam.read()
    k = cv2.waitKey(1)
    if k != 1:
 
       break
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faces = faceCascade.detectMultiScale(gray)
if len(faces) ==0:
    print("Failed to detect face")
    print("try taking off your mask")

for (x, y, w, h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0),2)
    print("face dectected")
    print("success")
cv2.imwrite('/home/pi/ex7/face.jpg', image)
cam.release()
cv2.destroyAllWindows()

