import cv2
import os

def capture_face():
    # Create faces directory if it doesn't exist
    if not os.path.exists('faces'):
        os.makedirs('faces')

    name = input("Enter employee name: ")
    # Clean the name to be filesystem safe
    name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).strip()
    
    if not name:
        print("Invalid name.")
        return

    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("Capture Face")
    
    img_counter = 0
    
    print(f"Press 'Space' to capture photo for {name}")
    print("Press 'ESC' to close")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        
        cv2.imshow("Capture Face", frame)
        
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"faces/{name}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1
            # We only need one good photo for this simple system, but loop allows retries or multiples if we expand
            print("Photo captured! Press ESC to finish or Space to overwrite/take another.")
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_face()
