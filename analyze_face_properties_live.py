import cv2
from deepface import DeepFace

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def load_models():
    try:
        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"
        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"

        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

    return faceNet, ageNet, genderNet

def process_frame(frame, faceNet, ageNet, genderNet, padding=20):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-18)', '(18-21)','(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        return resultImg

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]


        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            emotion = "N/A"
            print(f"Error in emotion detection: {e}")

        cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), 2)
        cv2.putText(resultImg, f'Gender: {gender}', (faceBox[0], faceBox[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Age: {age[1:-1]} years', (faceBox[0], faceBox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Emotion: {emotion}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

    return resultImg

def main():
    faceNet, ageNet, genderNet = load_models()
    if not faceNet or not ageNet or not genderNet:
        print("Failed to load models")
        return

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            print("Error: No frame captured.")
            break

        resultImg = process_frame(frame, faceNet, ageNet, genderNet)
        cv2.imshow('Real-time Age, Gender, and Emotion Detection', resultImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



