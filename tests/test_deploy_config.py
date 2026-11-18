"""Secretless checks of the Pages artifact and hosted connection policy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deploy.build_pages import build
from deploy.prepare_neon import owner_settings, pooled_url
from test_deployment import load_app


class DeployConfigTests(unittest.TestCase):
    def test_pages_contains_only_public_assets_and_https_link(self):
        origin = "https://cerviai-demo-example.us-central1.run.app"
        with TemporaryDirectory() as work:
            output = Path(work) / "pages"
            build(origin + "/", output)
            assets = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(assets, {"index.html", "static/style.css", "static/images/ai_doctor.png"})
            html = (output / "index.html").read_text()
            self.assertIn(f'href="{origin}/"', html)
            self.assertNotIn("__CERVIAI_APP_URL__", html)
            self.assertNotRegex(html, r"<form\b|<script\b|(?:src|href)=\"/")
            for bad in ("", "http://cerviai-demo.run.app", "https://cerviai.example.invalid",
                        "https://cerviai-demo.run.app/signin"):
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    build(bad, Path(work) / "rejected")
            self.assertFalse((Path(work) / "rejected").exists())

    def test_neon_setup_does_not_copy_owner_access(self):
        owner = "postgresql://neondb_owner:owner-dummy-value@ep-example.us-east-2.aws.neon.tech/neondb"
        app_url = pooled_url(owner_settings(owner), "fresh-dummy-app-password")
        self.assertEqual(app_url, (
            "postgresql://cerviai_app:fresh-dummy-app-password@ep-example-pooler.us-east-2.aws.neon.tech/"
            "neondb?sslmode=verify-full&channel_binding=require"
        ))
        with patch("psycopg.connect") as connect:
            with self.assertRaisesRegex(RuntimeError, "cerviai_app"):
                load_app(K_SERVICE="demo", DATABASE_URL=owner)
            connect.assert_not_called()

    def test_cloud_run_requires_verified_tls_even_with_a_default_uri(self):
        url = ("postgresql://cerviai_app:dummy-value@ep-example-pooler.us-east-2.aws.neon.tech/"
               "neondb?sslmode=require")
        with patch("psycopg.connect", side_effect=RuntimeError("offline DB stub")) as connect:
            with self.assertRaisesRegex(RuntimeError, "offline DB stub"):
                load_app(K_SERVICE="demo", DATABASE_URL=url)
            self.assertEqual(connect.call_args.kwargs["sslmode"], "verify-full")
            self.assertEqual(connect.call_args.kwargs["sslrootcert"], "system")
            self.assertEqual(connect.call_args.kwargs["channel_binding"], "require")


if __name__ == "__main__":
    unittest.main()
