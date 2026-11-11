
import cv2

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Check whether classifiers loaded successfully
if face_cascade.empty():
    raise IOError("Face cascade could not be loaded")

if smile_cascade.empty():
    raise IOError("Smile cascade could not be loaded")

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise IOError("Cannot access webcam")

print("Smart Selfie / Smile Detection Started")
print("Press Q to quit")

while True:

    # Capture webcam frame
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    smile_detected = False

    # Process every detected face
    for (x, y, w, h) in faces:

        # Extract face region
        face_gray = gray[y:y+h, x:x+w]

        # Detect smiles inside the face
        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 15)
        )

        if len(smiles) > 0:
            smile_detected = True

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        # Draw smile rectangles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(
                frame,
                (x+sx, y+sy),
                (x+sx+sw, y+sy+sh),
                (0, 255, 0),
                2
            )

    # Display smile indicator
    if smile_detected:
        message = "SMILE DETECTED!"
        color = (0, 255, 0)
    else:
        message = "NO SMILE"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        message,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # Display number of detected faces
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show webcam output
    cv2.imshow("Smart Selfie - Smile Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
camera.release()
cv2.destroyAllWindows()
