"""Secretless checks of the Pages artifact and hosted connection policy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deploy.build_pages import build
from deploy.prepare_neon import owner_settings, pooled_url
from test_deployment import HEADERS, load_app


OWNER_URL = "postgresql://neondb_owner:owner-dummy-value@ep-example.us-east-2.aws.neon.tech/neondb"
APP_URL = ("postgresql://cerviai_app:dummy-value@ep-example-pooler.us-east-2.aws.neon.tech/"
           "neondb?sslmode=disable&channel_binding=disable&hostaddr=127.0.0.1")


class DeployConfigTests(unittest.TestCase):
    def test_pages_contains_only_public_assets_and_https_link(self):
        origin = "https://cerviai-demo-example.up.railway.app"
        with TemporaryDirectory() as work:
            output = Path(work) / "pages"
            build(origin + "/", output)
            assets = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(assets, {"index.html", "static/style.css", "static/images/ai_doctor.png"})
            html = (output / "index.html").read_text()
            self.assertIn(f'href="{origin}/"', html)
            self.assertIn("Open demo on Railway", html)
            self.assertNotIn("__CERVIAI_APP_URL__", html)
            self.assertNotRegex(html, r"<form\b|<script\b|(?:src|href)=\"/")
            for bad in (
                "", "http://cerviai-demo.up.railway.app", "https://cerviai-demo.run.app", "https://railway.app",
                origin + ".example.invalid", origin + "/signin", origin + ":443", origin + "?foo=bar", origin + "#top",
                "https://user:pass@cerviai-demo-example.up.railway.app",
            ):
                with self.subTest(url=bad), self.assertRaisesRegex(ValueError, "HTTPS"):
                    build(bad, Path(work) / "rejected")
            self.assertFalse((Path(work) / "rejected").exists())

    def test_neon_setup_does_not_copy_owner_access(self):
        app_url = pooled_url(owner_settings(OWNER_URL), "fresh-dummy-app-password")
        self.assertEqual(app_url, (
            "postgresql://cerviai_app:fresh-dummy-app-password@ep-example-pooler.us-east-2.aws.neon.tech/"
            "neondb?sslmode=verify-full&channel_binding=require"
        ))
        with patch("psycopg.connect") as connect:
            with self.assertRaisesRegex(RuntimeError, "cerviai_app"):
                load_app(RAILWAY_PROJECT_ID="demo", DATABASE_URL=OWNER_URL)
            connect.assert_not_called()

    def test_railway_requires_all_runtime_settings(self):
        with patch("sqlite3.connect") as sqlite, patch("psycopg.connect") as postgres:
            for variable in ("DATABASE_URL", "SECRET_KEY", "DEMO_PASSWORD"):
                with self.subTest(variable=variable), self.assertRaisesRegex(RuntimeError, f"Configure .*{variable}"):
                    load_app(**{"RAILWAY_ENVIRONMENT_ID": "demo", "DATABASE_URL": APP_URL, variable: " "})
            sqlite.assert_not_called()
            postgres.assert_not_called()

    def test_railway_forces_verified_tls_and_sets_hosted_headers(self):
        with patch("psycopg.connect") as connect:
            connect.return_value.execute.return_value.fetchone.side_effect = [(False,), (True,)]
            module = load_app(RAILWAY_PROJECT_ID="demo", DATABASE_URL=APP_URL)
            self.addCleanup(module._media_directory.cleanup)
            connect.assert_called_once_with(
                host="ep-example-pooler.us-east-2.aws.neon.tech", user="cerviai_app", dbname="neondb",
                password="dummy-value", sslmode="verify-full", sslrootcert="system", channel_binding="require",
                connect_timeout=15, prepare_threshold=None,
            )
        client = module.app.test_client()
        origin = "https://cerviai-demo-example.up.railway.app"
        self.assertEqual(client.get("/healthz", base_url=origin).status_code, 200)
        self.assertEqual(client.get("/", base_url=origin).status_code, 401)
        form = client.get("/signin", base_url=origin, headers=HEADERS)
        self.assertEqual(form.status_code, 200)
        self.assertIn("frame-ancestors 'none'", form.headers["Content-Security-Policy"])
        for attribute in ("Secure", "HttpOnly", "SameSite=Lax"):
            self.assertIn(attribute, form.headers["Set-Cookie"])

    def test_railway_rejects_elevated_neon_role(self):
        with patch("psycopg.connect") as connect:
            connect.return_value.execute.return_value.fetchone.return_value = (True,)
            with self.assertRaisesRegex(RuntimeError, "restricted Neon role"):
                load_app(RAILWAY_ENVIRONMENT_ID="demo", DATABASE_URL=APP_URL)


if __name__ == "__main__":
    unittest.main()
