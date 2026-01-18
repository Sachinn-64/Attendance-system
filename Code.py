import cv2
import numpy as np
# pip install face-recognition # Install it if you do not have it.
import face_recognition
import os
from datetime import datetime
import pyttsx3
import tkinter as tk
from tkinter import simpledialog

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()


# The path where your faces images are stored.
# You have to store the person frontal image with his hame, this can be integrated with GUI that captures unknown face and ask you to name it something so it can store it with the name to recognize it later.
path = 'EmployeesFaces'  # change it according your project
if not os.path.exists(path):
    os.makedirs(path)

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print(f"No face found in one of the images.")
    return encodeList
 

marked_today_set = set()

def load_today_attendance():
    file_path = 'Attendance.csv'
    if not os.path.exists(file_path):
        return

    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    
    with open(file_path, 'r') as f:
        myDataList = f.readlines()
        
    for line in myDataList:
        entry = line.strip().split(',')
        if len(entry) >= 3:
            # entry[0] is Name, entry[2] is Date
            if entry[2] == dateString:
                marked_today_set.add(entry[0])
    print(f"Loaded attendance for today: {marked_today_set}")

def markAttendance(name):
    file_path = 'Attendance.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time,Date\n')

    # Removed the check for "marked_today" to log every instance
    
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dateString = now.strftime('%Y-%m-%d')
    
    with open(file_path, 'a') as f:
        f.write(f'{name},{dtString},{dateString}\n')
    
    # We can still use the set to avoid spamming the "Welcome" voice message every frame
    if name not in marked_today_set:
        marked_today_set.add(name)
        print(f"Attendance Marked: {name}")
        speak(f"Welcome {name}")
    else:
        # Just print to console to show it's logging without speaking
        print(f"Logging: {name} at {dtString}")

def register_new_user(img, path):
    # Create a hidden root window for the dialog
    root = tk.Tk()
    root.withdraw()
    
    # Ask for the name
    name = simpledialog.askstring("Register New User", "Enter Name:")
    root.destroy()
    
    if name:
        # Save the image
        filename = f"{path}/{name}.jpg"
        cv2.imwrite(filename, img)
        print(f"User {name} registered successfully!")
        speak(f"Registered {name}")
        return True
    return False

encodeListKnown = findEncodings(images)
print(f'Encoding Complete. Found {len(encodeListKnown)} known faces.')
load_today_attendance()


#This can be modified to work with, Static Images, Videos, Mobile/Laptop Cameras, RTSP streams of Security Camera/IOT Defices or Drones etc.
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera. Please check camera permissions.")
        break

    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    name = "Unknown"
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        if len(encodeListKnown) > 0:
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
            matchIndex = np.argmin(faceDis)
            
            if faceDis[matchIndex]< 0.50:
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else: name = 'Unknown'
        else:
            name = 'Unknown'
            
        #print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        
        # Color coding: Green for present, Red for unknown
        if name != "Unknown":
            color = (0, 255, 0) # Green
            cv2.putText(img, "Marked", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        else:
            color = (0, 0, 255) # Red

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),color,cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
    cv2.imshow('Webcam',img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        speak("Please look at the camera for registration")
        # Capture a clean frame for registration
        success, clean_img = cap.read()
        if success:
            if register_new_user(clean_img, path):
                # Reload known faces
                images = []
                classNames = []
                myList = os.listdir(path)
                for cl in myList:
                    curImg = cv2.imread(f'{path}/{cl}')
                    images.append(curImg)
                    classNames.append(os.path.splitext(cl)[0])
                encodeListKnown = findEncodings(images)
                print("Database updated!")

cap.release()
cv2.destroyAllWindows()
