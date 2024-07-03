import cv2
import os
import numpy as np
import faceRecognition as fr
import smtplib
from email.message import EmailMessage

test_img = cv2.imread('TestImages/frame76.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)



#Comment belows lines when running this program second time.Since it saves training.yml file in directory
faces,faceID=fr.labels_for_training_data('trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')


#Uncomment below line for subsequent runs one time traning image then uncomment to run this 2 line
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('trainingData.yml')


def deblur(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0,-1,0],
                    [-1, 5, -1],
                   [0,-1,0]])
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

deblurred_face = None


names = {0: "Riswanth", 1: "Dharneesh", 2: "Soundar", 3: "Mukul",4:"Rajesh"}

for face in faces_detected:
    
    (x, y, w, h) = face
    roi_gray = gray_img[y:y + h, x:x + h]
    
    is_blurry = cv2.Laplacian(roi_gray, cv2.CV_64F).var() < 100
    if is_blurry:
        print("Blurred face detected!")
        new=deblur(test_img)
        cv2.imshow('Orginal image', test_img)
        cv2.imshow('Deblurred Face', new)
        gray_img1 = fr.faceDetection(new)[1]
        roi_gray1 = gray_img1[y:y + h, x:x + w] 
        label, confidence = face_recognizer.predict(roi_gray1)

    else:
        label, confidence = face_recognizer.predict(roi_gray)

        
    
    print("confidence:", confidence)
    fr.draw_rect(test_img, face)
    cv2.imwrite('opencv.png', test_img)
    
    if confidence > 40:
        predicted_name = "Unknown"
        label="Unknown"
        
        # Sending an email for unknown face
        Sender_Email = "vmsoundarabharath@gmail.com"
        Recipient_Email = "vmsoundarabharath2003@gmail.com"
        Password = "oylc wozq lqzx jdkw"
        
        newMessage = EmailMessage()
        newMessage['Subject'] = "Alert: Unknown person detected"
        newMessage['From'] = Sender_Email
        newMessage['To'] = Recipient_Email
        
        # Attaching the image to the email
        with open('opencv.png', 'rb') as f:
            image_data = f.read()
            image_name = f.name  # File name without path
        newMessage.add_attachment(image_data, maintype='image', subtype='jpeg', filename=image_name)
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Sender_Email, Password)
            smtp.send_message(newMessage)
        
        print("Email sent successfully!")
    else:
        predicted_name = names.get(label, "Unknown")

    
    print("label:", label)
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (640, 480))
cv2.imshow("face detection tutorial", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
