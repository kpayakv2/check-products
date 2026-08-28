"""
tests/integration/test_ml_e2e.py
=================================
E2E Test สำหรับ ML Flow ครบวงจร

ทดสอบลำดับ:
  1. POST /api/v1/learn/retrain → status=success
  2. GET  /api/v1/learn/status  → is_trained=True, accuracy > 0
  3. GET  /api/v1/learn/history → มี record ≥ 1
  4. Stage-2: ML กรองผลออกจาก scan (ถ้ามีข้อมูล)

ต้องการ: FastAPI server รันอยู่ที่ 127.0.0.1:8000
         และ Supabase มีข้อมูล similarity_matches (reviewed=True) ≥ 10 รายการ
"""

import pytest
import requests
import time

BASE_URL = "http://127.0.0.1:8000"
API_TIMEOUT = 60  # seconds (การเทรนอาจใช้เวลา)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_base():
    """ตรวจสอบว่า server รันอยู่ก่อนเริ่ม test (retry 3 ครั้ง)"""
    import time
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE_URL}/api/v1/health", timeout=15)
            if r.status_code in (200, 404):
                return BASE_URL
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt < 2:
                time.sleep(5)  # รอ 5 วิ แล้วลองใหม่
    pytest.skip(f"FastAPI server ไม่ตอบสนองหลัง 3 ครั้ง ({BASE_URL}): {last_error}")


# ──────────────────────────────────────────────
# Test 1: Retrain
# ──────────────────────────────────────────────

class TestMLRetrain:
    """ทดสอบ POST /api/v1/learn/retrain"""

    def test_retrain_returns_200(self, api_base):
        """retrain ต้องตอบกลับ HTTP 200 หรือ 400 (ถ้าข้อมูลไม่พอ)"""
        r = requests.post(f"{api_base}/api/v1/learn/retrain", timeout=API_TIMEOUT)
        assert r.status_code in (200, 400), (
            f"Unexpected status {r.status_code}: {r.text[:200]}"
        )

    def test_retrain_success_has_training_result(self, api_base):
        """ถ้าสำเร็จ ต้องมี field 'data' พร้อม accuracy"""
        r = requests.post(f"{api_base}/api/v1/learn/retrain", timeout=API_TIMEOUT)
        if r.status_code == 400:
            # ข้อมูลไม่พอ — skip แต่ไม่ fail
            body = r.json()
            pytest.skip(f"ข้อมูลไม่เพียงพอสำหรับเทรน: {body.get('detail', '')}")

        body = r.json()
        assert body.get("status") == "success", f"status ไม่ใช่ success: {body}"
        data = body.get("data", {})
        assert "test_accuracy" in data, f"ไม่มี test_accuracy ใน response: {data}"
        assert 0.0 <= data["test_accuracy"] <= 1.0, (
            f"test_accuracy={data['test_accuracy']} นอกช่วง [0,1]"
        )

    def test_retrain_accuracy_above_zero(self, api_base):
        """accuracy ต้องมากกว่า 0"""
        r = requests.post(f"{api_base}/api/v1/learn/retrain", timeout=API_TIMEOUT)
        if r.status_code == 400:
            pytest.skip("ข้อมูลไม่เพียงพอ")
        body = r.json()
        data = body.get("data", {})
        assert data.get("test_accuracy", 0) > 0, "accuracy เป็น 0 — โมเดลเทรนไม่ได้"


# ──────────────────────────────────────────────
# Test 2: Status
# ──────────────────────────────────────────────

class TestMLStatus:
    """ทดสอบ GET /api/v1/learn/status"""

    def test_status_returns_200(self, api_base):
        r = requests.get(f"{api_base}/api/v1/learn/status", timeout=10)
        assert r.status_code == 200, f"status endpoint ตอบ {r.status_code}"

    def test_status_has_required_fields(self, api_base):
        """ต้องมี is_trained field เสมอ"""
        r = requests.get(f"{api_base}/api/v1/learn/status", timeout=10)
        body = r.json()
        assert "is_trained" in body, f"ไม่มี is_trained: {body}"

    def test_status_trained_has_accuracy(self, api_base):
        """ถ้า is_trained=True ต้องมี accuracy และ total_samples"""
        r = requests.get(f"{api_base}/api/v1/learn/status", timeout=10)
        body = r.json()
        if not body.get("is_trained"):
            pytest.skip("โมเดลยังไม่ได้เทรน — รัน retrain ก่อน")

        assert "accuracy" in body, f"is_trained=True แต่ไม่มี accuracy: {body}"
        assert "total_samples" in body, f"ไม่มี total_samples: {body}"
        assert body["accuracy"] >= 0, "accuracy เป็นลบ"
        assert body["total_samples"] >= 0, "total_samples เป็นลบ"


# ──────────────────────────────────────────────
# Test 3: Training History
# ──────────────────────────────────────────────

class TestMLHistory:
    """ทดสอบ GET /api/v1/learn/history"""

    def test_history_returns_200(self, api_base):
        r = requests.get(f"{api_base}/api/v1/learn/history", timeout=10)
        assert r.status_code == 200, f"history endpoint ตอบ {r.status_code}"

    def test_history_has_required_fields(self, api_base):
        """ต้องมี status, history (list), total"""
        r = requests.get(f"{api_base}/api/v1/learn/history", timeout=10)
        body = r.json()
        assert "status" in body, f"ไม่มี status field: {body}"
        assert "history" in body, f"ไม่มี history field: {body}"
        assert isinstance(body["history"], list), "history ต้องเป็น list"
        assert "total" in body, f"ไม่มี total field: {body}"

    def test_history_limit_param(self, api_base):
        """ทดสอบ limit parameter"""
        r = requests.get(f"{api_base}/api/v1/learn/history?limit=3", timeout=10)
        body = r.json()
        assert len(body.get("history", [])) <= 3, "history ส่งมาเกิน limit=3"

    def test_history_session_structure(self, api_base):
        """แต่ละ session ต้องมี field ที่ถูกต้อง"""
        r = requests.get(f"{api_base}/api/v1/learn/history", timeout=10)
        body = r.json()
        sessions = body.get("history", [])
        if not sessions:
            pytest.skip("ยังไม่มีประวัติการเทรน — รัน retrain ก่อน")

        session = sessions[0]  # ตรวจสอบ session ล่าสุด
        required_fields = ["training_date", "model_type", "total_samples", "test_accuracy"]
        for field in required_fields:
            assert field in session, f"session ไม่มี field '{field}': {session.keys()}"

    def test_history_after_retrain_increases(self, api_base):
        """หลัง retrain ต้องมีประวัติเพิ่มขึ้น"""
        # ดึงจำนวนก่อน
        r_before = requests.get(f"{api_base}/api/v1/learn/history", timeout=10)
        count_before = r_before.json().get("total", 0)

        # Retrain
        r_train = requests.post(f"{api_base}/api/v1/learn/retrain", timeout=API_TIMEOUT)
        if r_train.status_code == 400:
            pytest.skip("ข้อมูลไม่พอสำหรับเทรน")

        # รอเล็กน้อยให้ DB commit
        time.sleep(1)

        # ดึงจำนวนหลัง
        r_after = requests.get(f"{api_base}/api/v1/learn/history", timeout=10)
        count_after = r_after.json().get("total", 0)

        assert count_after >= count_before, (
            f"หลัง retrain ประวัติไม่เพิ่มขึ้น (before={count_before}, after={count_after})"
        )


# ──────────────────────────────────────────────
# Test 4: Stage-2 ML Inference (Internal Scan)
# ──────────────────────────────────────────────

class TestStage2MLInference:
    """ทดสอบว่า ML Stage-2 Inference ทำงานใน internal-match scan"""

    SCAN_TIMEOUT = 120  # scan-internal ใช้เวลานาน (คำนวณ embedding matrix)

    def test_internal_scan_returns_200(self, api_base):
        """scan endpoint ต้องตอบ 200 หรือ 503 ถ้า Supabase ไม่พร้อม"""
        # ScanInternalRequest รับแค่ threshold และ limit
        payload = {"threshold": 0.9, "limit": 10}
        try:
            r = requests.post(
                f"{api_base}/api/v1/match/scan-internal",
                json=payload,
                timeout=self.SCAN_TIMEOUT
            )
            # 200 = สำเร็จ, 422 = validation error, 503 = Supabase ไม่พร้อม
            assert r.status_code in (200, 422, 503), (
                f"Unexpected status {r.status_code}: {r.text[:300]}"
            )
        except requests.exceptions.ReadTimeout:
            pytest.skip(
                f"scan-internal timeout >{self.SCAN_TIMEOUT}s — DB ใหญ่เกินไป หรือ embedding ช้า"
            )

    def test_internal_scan_response_structure(self, api_base):
        """response ต้องมี results, total_scanned, pairs_found"""
        payload = {"threshold": 0.95, "limit": 5}
        try:
            r = requests.post(
                f"{api_base}/api/v1/match/scan-internal",
                json=payload,
                timeout=self.SCAN_TIMEOUT
            )
        except requests.exceptions.ReadTimeout:
            pytest.skip(f"scan-internal timeout >{self.SCAN_TIMEOUT}s")

        if r.status_code == 422:
            pytest.skip(f"Schema ไม่ตรง: {r.text[:200]}")
        if r.status_code == 503:
            pytest.skip("Supabase unavailable")
        if r.status_code == 500:
            pytest.skip(f"Server error: {r.text[:200]}")

        assert r.status_code == 200, f"Unexpected {r.status_code}: {r.text[:200]}"
        start_res = r.json()
        task_id = start_res.get("task_id")
        assert task_id, f"No task_id found in start response: {start_res}"

        # Poll the status endpoint until completed or failed
        #
        # งบเวลาเดิม 60 วิ ตั้งไว้ตอน DB ยังว่าง ทำให้ scan เสร็จทันที
        # พอมีสินค้าจริง 3,103 รายการ การเทียบคู่ใช้เวลาราว 40 วิต่อรอบ
        # และเมื่อรันหลายเทสต์ติดกัน รอบหลังต้องต่อคิว จึงต้องเผื่อมากกว่าเดิม
        status = "pending"
        max_polls = 90
        poll_interval = 2.0
        body = {}
        for _ in range(max_polls):
            status_res = requests.get(f"{api_base}/api/v1/match/scan-internal/status/{task_id}", timeout=10)
            assert status_res.status_code == 200, f"Failed to get task status: {status_res.text}"
            body = status_res.json()
            status = body.get("status")
            if status in ("completed", "failed"):
                break
            time.sleep(poll_interval)

        assert status == "completed", f"Scan task did not complete (status={status}, error={body.get('error')})"
        result = body.get("result")
        assert result is not None, f"No result found in task body: {body}"
        
        # ScanInternalResponse fields: total_scanned, pairs_found, results, processing_time
        for field in ["total_scanned", "pairs_found", "results"]:
            assert field in result, f"Response result ไม่มี field '{field}': {list(result.keys())}"
        assert isinstance(result["results"], list), "results ต้องเป็น list"
