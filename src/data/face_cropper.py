
# load used libraries
import cv2 # to reading images
import mediapipe as mp # to face detection


class FaceCropper:
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection = 1,
            min_detection_confidence = 0.5
        )

    def crop_face(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return None
        
        h, w, _ = image.shape
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        return image[y1:y2, x1:x2]
