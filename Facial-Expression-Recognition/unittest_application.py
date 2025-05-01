import unittest
from unittest.mock import patch, MagicMock
from application import app, preprocessing, store_emotion
import numpy as np
import torch


class TestApplication(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    # Test each rout
    def test_login_route(self):
        res = self.client.get('/login')
        self.assertEqual(res.status_code, 200)

    def test_create_session_route(self):
        res = self.client.get('/create_session')
        self.assertEqual(res.status_code, 200)

    def test_intensity_feed_route(self):
        res = self.client.get('/intensity_feed')
        self.assertEqual(res.status_code, 200)
        self.assertIn('multipart/x-mixed-replace', res.content_type)

    def test_video_feed_route(self):
        res = self.client.get('/video_feed')
        self.assertEqual(res.status_code, 200)
        self.assertIn('multipart/x-mixed-replace', res.content_type)

    @patch('application.session', {})
    def test_visualize_data_redirect_if_not_logged_in(self):
        res = self.client.get('/visualize_data')
        self.assertEqual(res.status_code, 302)  # Should redirect to login

    # test emotion recording
    def test_preprocessing_tensor_shape(self):
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = preprocessing(dummy_img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 1, 48, 48))

    @patch('application.psycopg2.connect')
    def test_store_emotion_empty_buffer(self, mock_conn):
        # Should not attempt DB insert with empty buffer
        store_emotion(session_id=1, intensities_buffer=[])
        mock_conn.assert_not_called()

    @patch('application.psycopg2.connect')
    def test_store_emotion_valid_buffer(self, mock_connect):
        buffer = [[0.1] * 7 for _ in range(5)]  # 5-frame buffer of 7 emotions
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        store_emotion(session_id=123, intensities_buffer=buffer)

        mock_connect.assert_called_once()
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
