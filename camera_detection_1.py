import cv2

cap = cv2.VideoCapture(0)

# Try disabling auto-focus and auto-exposure (might not work for all webcams)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
cap.set(cv2.CAP_PROP_FOCUS, 50)  # Adjust focus (try different values)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Set manual exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust exposure (try different values)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
