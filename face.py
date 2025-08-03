from fer import FER
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

detector = FER(mtcnn=False)  # Try True also for comparison

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally for natural view

    if not ret:
        print("Error: Failed to grab frame")
        break

    result = detector.detect_emotions(frame)

    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, top_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Facial Expression Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
