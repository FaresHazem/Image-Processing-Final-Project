import cv2
import numpy as np
import os

# Load a sample image
sample_image_path = r'C:\Users\fares\Desktop\Lane_Detection_Sliding_Windows-main\image.jpg'

# Check if the path is correct
print("Sample image path:", sample_image_path)
print("Image exists:", os.path.exists(sample_image_path))

# Load the image
sample_image = cv2.imread(sample_image_path)
if sample_image is None:
    print("Failed to load the image. Check the file path.")
else:
    # Resize image for processing
    frame = cv2.resize(sample_image, (640, 480))

    # Convert the image to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Function to handle slider changes
    def nothing(x):
        pass

    # Create a window for the trackbars
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

    while True:
        # Get current positions of trackbars
        l_h = cv2.getTrackbarPos("L-H", "Trackbars")
        l_s = cv2.getTrackbarPos("L-S", "Trackbars")
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        u_h = cv2.getTrackbarPos("U-H", "Trackbars")
        u_s = cv2.getTrackbarPos("U-S", "Trackbars")
        u_v = cv2.getTrackbarPos("U-V", "Trackbars")

        # Define HSV range for thresholding
        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])

        # Threshold the HSV image
        mask = cv2.inRange(hsv_frame, lower, upper)

        # Show the original image and the thresholded image
        cv2.imshow("Original Image", frame)
        cv2.imshow("Thresholded Image", mask)

        # Break loop on pressing 'Esc'
        if cv2.waitKey(1) == 27:
            break

    # Cleanup
    cv2.destroyAllWindows()
