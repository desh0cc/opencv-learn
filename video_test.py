import cv2 as cv
import numpy as np

def callback(input):
    pass

def video_test():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        exit()


    winname = 'cap'
    cv.namedWindow(winname)
    cv.createTrackbar('mint',winname,100,255,callback)
    cv.createTrackbar('maxt',winname,150,255,callback)
    cv.createTrackbar('thresh',winname,150,255,callback)


    while True:
        ret,frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(frame,(3,3),3)

            mint = cv.getTrackbarPos('mint',winname)
            maxt = cv.getTrackbarPos('maxt',winname)
            thresh = cv.getTrackbarPos('thresh',winname)

            filtered = cv.Canny(blurred,mint,maxt)

            dist = 1
            angle = np.pi/180
            threshold = thresh
            lines = cv.HoughLines(filtered,dist,angle,threshold)

            k = 3000

            if lines is not None:
                for line in lines:
                    rho,theta = line[0] # type: ignore
                    dhat = np.array([[np.cos(theta)],[np.sin(theta)]])
                    lhat =  np.array([[-np.sin(theta)],[np.cos(theta)]])

                    d = rho*dhat

                    p1 = d + k*lhat
                    p2 = d - k*lhat

                    p1 = p1.astype(int)
                    p2 = p2.astype(int)

                    cv.line(filtered,(p1[0][0],p1[1][0]), (p2[0][0],p2[1][0]),(255,255,255),2)
            
            cv.imshow(winname,filtered)
            
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def sift_window():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        ret,frame = cap.read()

        if cv.waitKey(1) == ord('q'):
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT.create()
        keyps = sift.detect(gray,None)

        gray = cv.drawKeypoints(gray,keyps,gray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv.imshow('sift',gray)


    cap.release()
    cv.destroyAllWindows()



sift_window()
# video_test()