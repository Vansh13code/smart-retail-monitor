import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

class TestSmartRetailEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_image_path = os.path.join(os.path.dirname(__file__), "img1.jpg")
        
        # Ensure tests/img1.jpg exists or create a placeholder for testing
        if not os.path.exists(cls.test_image_path):
            import numpy as np
            import cv2
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Test Image", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(cls.test_image_path, placeholder)

    def test_01_health(self):
        """Test health check endpoint"""
        response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data.get("status"), "ok")

    def test_02_upload_image(self):
        """Test image upload endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/upload-image/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertEqual(res_json["data"]["file_type"], "image")
        self.assertIn("filename", res_json["data"])

    def test_03_detect_shelf_file(self):
        """Test shelf detection using raw file upload"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/detect-shelf/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("detections", res_json["data"])

    def test_04_detect_shelf_filename(self):
        """Test shelf detection using pre-uploaded filename"""
        # Upload first
        with open(self.test_image_path, "rb") as f:
            self.client.post("/upload-image/", files={"file": ("test_shared.jpg", f, "image/jpeg")})
            
        # Call with filename
        response = self.client.post(
            "/detect-shelf/",
            data={"filename": "test_shared.jpg"}
        )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("detections", res_json["data"])

    def test_05_detect_products(self):
        """Test product detection endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/detect-products/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("result", res_json["data"])

    def test_06_classify_products(self):
        """Test product classification endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/classify-products/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("classified_products", res_json["data"]["result"])

    def test_07_inventory_analysis(self):
        """Test inventory analysis endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/inventory-analysis/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIsInstance(res_json["data"]["result"], list)

    def test_08_ocr(self):
        """Test price tag OCR endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/ocr/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("detections", res_json["data"]["result"])

    def test_09_customer_analysis(self):
        """Test customer tracking endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/customer-analysis/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("customer_count", res_json["data"]["result"])

    def test_10_complete_pipeline(self):
        """Test complete pipeline execution"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/complete-pipeline/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("detections", res_json["data"]["result"])

    def test_11_generate_report(self):
        """Test report generation endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post(
                "/generate-report/",
                files={"file": ("test_img.jpg", f, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("report", res_json["data"])

    def test_12_video_analysis_start(self):
        """Test starting a video background task"""
        # Create a mock short video for tests
        test_video_path = os.path.join(os.path.dirname(__file__), "mock_test.mp4")
        if not os.path.exists(test_video_path):
            import numpy as np
            import cv2
            out = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
            for i in range(15):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                cv2.putText(frame, str(i), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                out.write(frame)
            out.release()

        with open(test_video_path, "rb") as f:
            response = self.client.post(
                "/video-analysis/",
                files={"file": ("mock_test.mp4", f, "video/mp4")}
            )
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertTrue(res_json.get("success"))
        self.assertIn("task_id", res_json["data"])
        
        # Clean up mock video
        try:
            os.remove(test_video_path)
        except Exception:
            pass

if __name__ == "__main__":
    unittest.main()
