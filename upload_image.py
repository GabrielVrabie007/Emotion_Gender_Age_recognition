import os
from tkinter import Tk, filedialog, messagebox
import cv2
from analyze_face_properties_live import load_models, process_frame

def upload_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if file_path:
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return cv2.imread(file_path)
        else:
            messagebox.showerror("Error", "Please select a valid image file.")
            return None

def main_func():
    faceNet, ageNet, genderNet = load_models()
    while True:
        frame = upload_image()
        if frame is None:
            break
        resultImg = process_frame(frame, faceNet, ageNet, genderNet)
        cv2.imshow('Age, Gender, and Emotion Detection', resultImg)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_func()

