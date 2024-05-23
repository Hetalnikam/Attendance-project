# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# # Load the trained face recognition model
# model = load_model('face_recognition_model.h5')

# # Load the Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         face_roi = frame[y:y+h, x:x+w]
#         face_roi = cv2.resize(face_roi, (100, 100))
#         face_roi = np.expand_dims(face_roi, axis=0) / 255.0

#         # Perform face recognition using the trained model
#         prediction = model.predict(face_roi)
#         predicted_label = np.argmax(prediction)

#         # Display the recognized face label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(frame, f'Person {predicted_label + 1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

#     # Display the video feed using matplotlib
#     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained face recognition model
model = load_model('face_recognition_model.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        face_roi = np.expand_dims(face_roi, axis=0) / 255.0

        # Perform face recognition using the trained model
        prediction = model.predict(face_roi)
        predicted_label = np.argmax(prediction)

        # Display the recognized face label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Person {predicted_label + 1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the video feed using matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Check for keyboard interrupt 'q' to stop the code
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
