from tkinter import *
from PIL import Image, ImageTk
import Test
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

mutex=1

def takeAttendance(mutex):
    path = 'imagesAttendance'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)  

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def markAttendance(name):
        with open('Attendance.csv','+r') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

    encodeListKnown = findEncodings(images)
    print('Encoding complete')

    cap = cv2.VideoCapture(0)

    while mutex:
        success,img = cap.read()
        # imgS = cv2.resize(img , (0,0) , 0.25 , 0.25 )
        imgN = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgN)
        encodesCurFrame = face_recognition.face_encodings(imgN,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)


        cv2.imshow('Webcam',img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()


# def stopAttendance():
#     mutex = 0

def startAttendance():
    screen = Tk()
    screen.title("Student Attendance")
    screen.geometry("700x475")
    screen.resizable(True, True)
    screen.configure(background='#9c8652')

    Button(screen, text='Start Attendance',command= lambda: takeAttendance(mutex), font=(
        'Verdana', 25)).place(x=75, y=200, width=250, height=75)

    Button(screen, text='Stop Attendance',command= lambda: takeAttendance(0), font=(
        'Verdana', 25)).place(x=375, y=200, width=250, height=75)


window = Tk()
window.title("Face Recognition Based Attendance System")
window.geometry("1200x750")
window.resizable(True, True)
window.configure(background='#355454')

# Creating a photoimage object to use image
img1 = Image.open("student-login.png")
resize_img1 = img1.resize((300, 300))

img2 = Image.open("admin-login.png")
resize_img2 = img2.resize((300, 300))

photoS = ImageTk.PhotoImage(resize_img1)
photoA = ImageTk.PhotoImage(resize_img2)

Button(window, text='Click Me !', image=photoS,
       command=startAttendance).place(x=200, y=200)
Button(window, text='Student', font=(
    'Verdana', 15)).place(x=200, y=505, width=308, height=50)

Button(window, text='Click Me !', image=photoA).place(x=700, y=200)
Button(window, text='Admin', font=(
    'Verdana', 15)).place(x=700, y=505, width=308, height=50)

window.mainloop()
