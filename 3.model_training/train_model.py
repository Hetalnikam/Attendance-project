# import os
# import numpy as np
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.optimizers import Adam

# # Load preprocessed images and labels
# data_dir = 'preprocessed_dataset'
# X_train = []
# y_train = []

# for img_name in os.listdir(data_dir):
#     img_path = os.path.join(data_dir, img_name)
#     img = cv2.imread(img_path)
    
#     if img is not None:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#         img = img / 255.0  # Normalize pixel values
#         X_train.append(img)
#         label = int(img_name.split('_')[1])  # Extract label from image filename
#         y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # Define the CNN model architecture
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(5, activation='softmax'))  # Assuming 5 classes (5 individuals)

# # Compile the model
# model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# # Save the trained model
# model.save('face_recognition_model.h5')
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
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
