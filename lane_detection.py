import cv2 as cv
import processes as ps

record_video = cv.VideoCapture('test.mp4')
while record_video.isOpened():
    rt, frames = record_video.read()
    if rt == False:
        record_video = cv.VideoCapture('test.mp4')
        continue
    detected_video = ps.final_process(frames)
    cv.imshow('frame', detected_video)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
record_video.release()
cv.destroyAllWindows()
