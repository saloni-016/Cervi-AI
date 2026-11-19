"""Offline deployment smoke tests. Model stubs never use patient data or Groq."""

import importlib.util
import io
import os
import re
import sys
import unittest
from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

from PIL import Image
from werkzeug.security import check_password_hash


ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "separate-app-test-password"
HEADERS = {"Authorization": "Basic " + b64encode(b"demo:local-smoke-invitation").decode()}
FORM = dict.fromkeys((
    "Age", "Number_of_sexual_partners", "First_sexual_intercourse", "Smokes",
    "Smokes_years", "Hormonal_Contraceptives", "IUD", "STDs", "STDs_number",
    "Dx_Cancer", "Dx_CIN", "Dx_HPV",
), "0")
FORM.update(Age="30", Number_of_sexual_partners="1", First_sexual_intercourse="20")


def stub_explain_image(original, heatmap):
    with Image.open(original) as image:
        image.save(heatmap, format="PNG")
    return {
        "predicted_cell_type": "Metaplastic",
        "confidence": 85.0,
        "abnormality_score": 8.0,
        "risk_indicator": "Low",
        "recommendation": "Educational demo",
        "explained_target": "Metaplastic",
        "top_regions": [],
        "disclaimer": "Not a diagnosis",
    }


def load_app(**settings):
    clinical = ModuleType("explain_xgboost")
    clinical.explain_clinical_input = lambda _: {
        "probability": 42.5, "prediction": 0, "explanations": [],
    }
    clinical.create_clinical_explanation = lambda _: "Test explanation"
    vit = ModuleType("explain_vit")
    vit.explain_image = stub_explain_image
    env = {
        "SECRET_KEY": "session-key-for-offline-test-only",
        "DEMO_PASSWORD": "local-smoke-invitation",
        "K_SERVICE": "",
        "RAILWAY_PROJECT_ID": "",
        "RAILWAY_ENVIRONMENT_ID": "",
        "DATABASE_URL": "",
        "GROQ_API_KEY": "",
        **settings,
    }
    spec = importlib.util.spec_from_file_location("cerviai_test_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    stubs = {"explain_xgboost": clinical, "explain_vit": vit, spec.name: module}
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        with patch.dict(os.environ, env), patch("dotenv.load_dotenv"):
            spec.loader.exec_module(module)
    finally:
        for name, original in previous.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original
    module.app.config["TESTING"] = True
    return module


def demo_image():
    stream = io.BytesIO()
    Image.new("RGB", (4, 4)).save(stream, format="PNG")
    stream.seek(0)
    return stream, "synthetic.png"


class DeploymentSmokeTests(unittest.TestCase):
    database_url = ""  # Override only in tests using a disposable PostgreSQL database.

    def setUp(self):
        workdir = TemporaryDirectory(prefix="cerviai-test-")
        self.addCleanup(workdir.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(workdir.name)
        self.module = load_app(DATABASE_URL=self.database_url)
        self.addCleanup(self.module._media_directory.cleanup)
        self.app = self.module.app
        self.client = self.app.test_client()
        self.email = f"{uuid4().hex}@example.invalid"

    def form_token(self, client, path):
        form = client.get(path, headers=HEADERS)
        self.assertEqual(form.status_code, 200, path)
        token = re.search(rb'name="csrf_token" value="([^"]+)"', form.data)
        self.assertIsNotNone(token, path)
        return token[1].decode()

    def post_form(self, client, endpoint, form_page, data, **kwargs):
        token = self.form_token(client, form_page)
        return client.post(endpoint, headers=HEADERS, data={**data, "csrf_token": token}, **kwargs)

    def signup(self, client):
        email = self.email if client is self.client else f"{uuid4().hex}@example.invalid"
        return self.post_form(client, "/signup", "/signup", {
            "username": "Demo tester", "email": email,
            "password": PASSWORD, "confirm_password": PASSWORD,
        })

    def test_invitation_and_chat_fallback(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        denied = self.client.get("/")
        self.assertEqual(denied.status_code, 401)
        self.assertIn("Basic", denied.headers["WWW-Authenticate"])
        landing = self.client.get("/", headers=HEADERS)
        self.assertEqual(landing.status_code, 200)
        self.assertEqual(landing.headers["Cache-Control"], "private, no-store")
        chat = "/clinical-result-chat"
        self.assertEqual(self.client.post(chat, headers=HEADERS, json={"question": "Hi"}).status_code, 401)
        self.assertEqual(self.signup(self.client).location, "/choose")
        signed_in = self.client.get("/choose", headers=HEADERS)
        token = re.search(rb'data-csrf-token="([^"]+)"', signed_in.data)[1].decode()
        chat_headers = {**HEADERS, "X-CSRF-Token": token}
        self.assertEqual(self.client.post(chat, headers=HEADERS, json={"question": "Hi"}).status_code, 403)
        self.assertEqual(self.client.post(chat, headers=chat_headers, json={"question": "Hi"}).status_code, 200)
        answer = self.client.post(chat, headers=chat_headers, json={"question": "Explain", "result_context": {}})
        self.assertEqual(answer.status_code, 503)
        self.assertIn("not configured", answer.json["answer"])

    def test_cloud_run_refuses_missing_database(self):
        with patch("sqlite3.connect", side_effect=AssertionError("must not fall back to SQLite")):
            with self.assertRaisesRegex(RuntimeError, "Configure DATABASE_URL"):
                load_app(K_SERVICE="demo", DATABASE_URL="")

    def test_account_and_history_persistence(self):
        self.assertEqual(self.signup(self.client).location, "/choose")
        self.assertIn(b"Email already registered", self.signup(self.client).data)
        self.assertEqual(self.client.post("/logout", headers=HEADERS).status_code, 403)
        self.assertEqual(self.post_form(self.client, "/logout", "/logout", {}).location, "/")
        self.assertEqual(self.post_form(self.client, "/signin", "/signin", {
            "email": self.email, "password": "wrong-password",
        }).status_code, 200)
        self.assertEqual(self.post_form(self.client, "/signin", "/signin", {
            "email": self.email, "password": PASSWORD,
        }).location, "/choose")

        clinical = self.post_form(self.client, "/predict", "/details", FORM, follow_redirects=True)
        self.assertEqual(clinical.status_code, 200)
        self.assertEqual(self.client.get("/result", headers=HEADERS).status_code, 200)
        combined = self.post_form(self.client, "/predict_combined", "/combined_assessment",
                                  {**FORM, "scan_image": demo_image()}, follow_redirects=True)
        self.assertEqual(combined.status_code, 200)
        self.assertIn(b"interpretation is currently unavailable", combined.data)
        self.assertIn(b"42.5", self.client.get("/history", headers=HEADERS).data)
        with self.client.session_transaction() as session:
            self.assertRegex(session["last_result"], r"^[0-9a-f]{32}$")
            self.assertRegex(session["last_combined_result"], r"^[0-9a-f]{32}$")
            self.assertNotIn("prob_percent", session)
        with self.module.database() as db:
            query = self.module.db_statement
            hashed = db.execute(query("SELECT password FROM users WHERE email=?"), (self.email,)).fetchone()[0]
            self.assertNotEqual(hashed, PASSWORD)
            self.assertTrue(check_password_hash(hashed, PASSWORD))
            rows = db.execute(query("SELECT probability FROM results WHERE email=?"), (self.email,)).fetchall()
            self.assertEqual(rows, [(42.5,), (42.5,)])
        self.assertEqual(Path("database.db").exists(), not bool(self.database_url))

    def test_media_requires_the_uploaders_session(self):
        self.signup(self.client)
        upload = self.post_form(self.client, "/predict_image", "/image_upload", {"scan_image": demo_image()})
        self.assertEqual(upload.location, "/image_result")
        self.assertEqual(self.client.get("/image_result", headers=HEADERS).status_code, 200)
        self.assertEqual(self.client.get("/full_report", headers=HEADERS).status_code, 200)
        result = self.client.get("/api/image_result", headers=HEADERS).json
        other = self.app.test_client()
        self.signup(other)
        for url in (result["image_url"], result["explanation_url"]):
            with self.client.get(url, headers=HEADERS) as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["Cache-Control"], "private, no-store")
            self.assertEqual(other.get(url, headers=HEADERS).status_code, 404)
        bad = self.post_form(self.client, "/predict_image", "/image_upload",
                             {"scan_image": (io.BytesIO(b"not an image"), "invalid.png")})
        self.assertEqual(bad.status_code, 400)
        with patch.object(self.module, "MAX_MEDIA_SETS", 1):
            self.post_form(self.client, "/predict_image", "/image_upload", {"scan_image": demo_image()})
        self.assertEqual(len(list(self.module.MEDIA_ROOT.iterdir())), 1)
        self.assertEqual(self.client.get(result["image_url"], headers=HEADERS).status_code, 404)


if __name__ == "__main__":
    unittest.main()
