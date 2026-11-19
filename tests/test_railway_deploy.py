"""Exercise the Railway deployment job with a fake GraphQL transport."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deploy import deploy_railway as railway


PROJECT = "11111111-1111-4111-8111-111111111111"
ENVIRONMENT = "22222222-2222-4222-8222-222222222222"
SERVICE = "33333333-3333-4333-8333-333333333333"
DEPLOYMENT = "44444444-4444-4444-8444-444444444444"
REPO = "saloni-016/Cervi-AI"
SHA = "f" * 40
PACKAGE = f"ghcr.io/{REPO.lower()}"
TAG = f"{PACKAGE}:{SHA}-12-1"
HOST = "cerviai-demo-example.up.railway.app"
ORIGIN = f"https://{HOST}"
TOKEN = "fake-token-for-offline-tests"
SCOPE = {"projectId": PROJECT, "environmentId": ENVIRONMENT}
CONFIG = {
    "healthcheckPath": "/healthz", "numReplicas": 1, "startCommand": None,
    "source": {"repo": None, "image": PACKAGE + ":previous-build"},
    "domains": {"serviceDomains": [{"domain": HOST}]},
}
TARGET = {"serviceId": SERVICE, "environmentId": ENVIRONMENT}
SETTINGS = {
    "RAILWAY_PROJECT_ID": PROJECT, "RAILWAY_ENVIRONMENT_ID": ENVIRONMENT, "RAILWAY_SERVICE_ID": SERVICE,
    "RAILWAY_PROJECT_TOKEN": TOKEN, "CERVIAI_APP_URL": ORIGIN + "/", "CERVIAI_IMAGE_TAG": TAG,
    "GITHUB_REPOSITORY": REPO, "GITHUB_SHA": SHA, "GITHUB_RUN_ID": "12", "GITHUB_RUN_ATTEMPT": "2",
}


def response(**fields):
    return io.BytesIO(json.dumps({"data": fields}).encode())


class RailwayDeployTests(unittest.TestCase):
    def setUp(self):
        self.http = self.enterContext(patch.object(railway.OPENER, "open"))
        self.pause = self.enterContext(patch.object(railway, "sleep"))
        self.enterContext(redirect_stdout(io.StringIO()))

    def start_deployment(self, *statuses):
        self.http.side_effect = [
            response(projectToken=SCOPE), response(serviceInstance=CONFIG), response(serviceInstanceUpdate=True),
            response(serviceInstanceDeployV2=DEPLOYMENT),
            *(response(deployment={"id": DEPLOYMENT, "status": status}) for status in statuses),
        ]

    def deploy(self):
        return railway.deploy(TOKEN, PROJECT, ENVIRONMENT, SERVICE, TAG, ORIGIN)

    def test_rollout_uses_tag_from_this_run_and_waits_for_new_deployment(self):
        self.start_deployment("QUEUED", "DEPLOYING", "SUCCESS")
        with TemporaryDirectory() as work:
            output = Path(work) / "github-output"
            with patch.dict(os.environ, {**SETTINGS, "GITHUB_OUTPUT": str(output)}, clear=True):
                railway.main()
            self.assertEqual(output.read_text(), f"id={DEPLOYMENT}\nurl={ORIGIN}\n")
        requests = [call.args[0] for call in self.http.call_args_list]
        for request in requests:
            self.assertEqual(request.full_url, railway.API_URL)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Project-access-token"), TOKEN)
            self.assertIsNone(request.get_header("Authorization"))
        requests = [json.loads(request.data) for request in requests]
        self.assertEqual([req["query"] for req in requests], [
            railway.PROJECT_SCOPE, railway.SERVICE_SETTINGS, railway.UPDATE_IMAGE, railway.DEPLOY_IMAGE,
            *([railway.DEPLOYMENT_STATUS] * 3),
        ])
        self.assertEqual(requests[2]["variables"], {**TARGET, "input": {"source": {"image": TAG}}})
        self.assertEqual(requests[3]["variables"], TARGET)
        self.assertTrue(all(req["variables"] == {"id": DEPLOYMENT} for req in requests[4:]))
        self.assertEqual(self.pause.call_count, 2)

    def test_rejects_wrong_token_scope_before_any_write(self):
        self.http.side_effect = [response(projectToken={**SCOPE, "environmentId": "another-env"})]
        with self.assertRaisesRegex(railway.DeployError, "does not match"):
            self.deploy()
        self.http.assert_called_once()

    def test_rejects_invalid_service_configuration_before_any_write(self):
        for changes, message in (
            ({"source": {"repo": REPO, "image": None}}, "GHCR"),
            ({"source": {"image": "ghcr.io/other/app:latest"}}, "GHCR"),
            ({"healthcheckPath": None}, "/healthz"),
            ({"startCommand": "flask run"}, "Start Command"),
            ({"numReplicas": 2}, "one Railway replica"),
            ({"domains": {"serviceDomains": []}}, "CERVIAI_APP_URL"),
        ):
            with self.subTest(changes=changes):
                self.http.reset_mock()
                self.http.side_effect = [response(projectToken=SCOPE), response(serviceInstance={**CONFIG, **changes})]
                with self.assertRaisesRegex(railway.DeployError, message):
                    self.deploy()
                self.assertEqual(self.http.call_count, 2)

    def test_checks_graphql_errors_without_logging_response(self):
        self.http.side_effect = [
            response(projectToken=SCOPE), response(serviceInstance=CONFIG),
            io.BytesIO(json.dumps({"data": {"serviceInstanceUpdate": True}, "errors": [{"message": TOKEN}]}).encode()),
        ]
        with self.assertRaisesRegex(railway.DeployError, "rejected serviceInstanceUpdate") as error:
            self.deploy()
        self.assertNotIn(TOKEN, str(error.exception))
        self.assertEqual(self.http.call_count, 3)

    def test_reports_failed_deployment_and_timeout(self):
        self.start_deployment("CRASHED")
        with self.assertRaisesRegex(railway.DeployError, f"{DEPLOYMENT} is CRASHED"):
            self.deploy()
        self.start_deployment("DEPLOYING")
        with patch.object(railway, "monotonic", side_effect=[0, 1, railway.DEPLOY_TIMEOUT + 1]):
            with self.assertRaisesRegex(railway.DeployError, f"Timed out waiting for deployment {DEPLOYMENT}"):
                self.deploy()

    def test_rejects_missing_token_and_non_published_tag(self):
        for change, message in (
            ({"RAILWAY_PROJECT_TOKEN": ""}, "Missing RAILWAY_PROJECT_TOKEN"),
            ({"CERVIAI_IMAGE_TAG": PACKAGE + ":latest"}, "unique GHCR tag"),
            ({"CERVIAI_IMAGE_TAG": f"{PACKAGE}:{SHA}-11-1"}, "unique GHCR tag"),
        ):
            with patch.dict(os.environ, {**SETTINGS, **change}, clear=True), self.assertRaisesRegex(SystemExit, message):
                railway.main()
        self.http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
