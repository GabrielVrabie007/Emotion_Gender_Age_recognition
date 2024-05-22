import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from analyze_face_properties_live import highlightFace


class TestFaceDetection(unittest.TestCase):

    @patch('cv2.dnn.readNet')
    def setUp(self, mock_readNet):
        self.mock_faceNet = MagicMock()
        self.mock_ageNet = MagicMock()
        self.mock_genderNet = MagicMock()
        mock_readNet.side_effect = [self.mock_faceNet, self.mock_ageNet, self.mock_genderNet]

        self.frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        self.blob = np.random.rand(1, 3, 300, 300).astype(np.float32)

        self.mock_detections = np.array([[[[0, 0, 0.9, 0.1, 0.1, 0.5, 0.5]]]])

        self.mock_faceNet.forward.return_value = self.mock_detections

    def test_highlightFace(self):
        with patch('cv2.dnn.blobFromImage', return_value=self.blob):
            resultImg, faceBoxes = highlightFace(self.mock_faceNet, self.frame, conf_threshold=0.7)

            self.assertEqual(len(faceBoxes), 1)
            self.assertEqual(faceBoxes[0], [30, 30, 150, 150])

            self.assertEqual(resultImg.shape, self.frame.shape)

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey', return_value=ord('q'))
    @patch('DeepFace.analyze')
    def test_main_logic(self, mock_analyze, mock_waitKey, mock_imshow, mock_VideoCapture):
        mock_video = MagicMock()
        mock_VideoCapture.return_value = mock_video
        mock_video.read.return_value = (True, self.frame)

        mock_analyze.return_value = [{'dominant_emotion': 'happy'}]

        self.assertTrue(mock_analyze.called)

        self.assertTrue(mock_imshow.called)

        self.assertTrue(mock_waitKey.called)


if __name__ == '__main__':
    unittest.main()
