"""Deploy the current GitHub Actions run's tested GHCR image to Railway."""

import json
import os
import re
import sys
from http.client import HTTPException
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .build_pages import railway_origin


API_URL = "https://backboard.railway.com/graphql/v2"
ID_PATTERN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
POLL_INTERVAL = 10
DEPLOY_TIMEOUT = 12 * 60
PENDING_STATES = ("INITIALIZING", "QUEUED", "WAITING", "BUILDING", "DEPLOYING")
FAILED_STATES = ("FAILED", "CRASHED", "REMOVING", "REMOVED", "SKIPPED", "SLEEPING", "NEEDS_APPROVAL")

PROJECT_SCOPE = "query { projectToken { projectId environmentId } }"
SERVICE_SETTINGS = """
query ($serviceId: String!, $environmentId: String!) {
  serviceInstance(serviceId: $serviceId, environmentId: $environmentId) {
    healthcheckPath numReplicas startCommand source { image repo }
    domains { serviceDomains { domain } }
  }
}
"""
UPDATE_IMAGE = """
mutation ($serviceId: String!, $environmentId: String!, $input: ServiceInstanceUpdateInput!) {
  serviceInstanceUpdate(serviceId: $serviceId, environmentId: $environmentId, input: $input)
}
"""
DEPLOY_IMAGE = """
mutation ($serviceId: String!, $environmentId: String!) {
  serviceInstanceDeployV2(serviceId: $serviceId, environmentId: $environmentId)
}
"""
DEPLOYMENT_STATUS = "query ($id: String!) { deployment(id: $id) { id status } }"


class DeployError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the project token to another URL.


OPENER = build_opener(NoRedirect())


def graphql(token, field, query, **variables):
    request = Request(API_URL, json.dumps({"query": query, "variables": variables}).encode(), headers={
        "Content-Type": "application/json",
        "Project-Access-Token": token,
    })
    # Do not dump response bodies or retry mutations after an ambiguous failure.
    try:
        with OPENER.open(request, timeout=20) as response:
            result = json.load(response)
    except HTTPError as error:
        error.close()
        raise DeployError(f"Railway API returned HTTP {error.code} for {field}.") from None
    except (OSError, HTTPException, ValueError):
        raise DeployError(f"Could not read the Railway API response for {field}.") from None
    data = result.get("data") if isinstance(result, dict) and not result.get("errors") else None
    if not isinstance(data, dict) or data.get(field) is None:
        raise DeployError(f"Railway rejected {field}. Check the project token, target IDs and service settings.")
    return data[field]


def wait_for_deployment(token, deployment_id):
    deadline = monotonic() + DEPLOY_TIMEOUT
    last_status = None
    while monotonic() < deadline:
        deployment = graphql(token, "deployment", DEPLOYMENT_STATUS, id=deployment_id)
        if deployment.get("id") != deployment_id:
            raise DeployError("Railway returned a different deployment ID.")
        status = deployment.get("status")
        if status == "SUCCESS":
            return
        if status in FAILED_STATES:
            raise DeployError(f"Deployment {deployment_id} is {status}. Check Railway Deployments and logs.")
        if status not in PENDING_STATES:
            raise DeployError(f"Deployment {deployment_id} returned an unrecognized status. Check Railway Deployments.")
        if status != last_status:
            print(f"Railway deployment {deployment_id}: {status}")
            last_status = status
        sleep(POLL_INTERVAL)
    raise DeployError(f"Timed out waiting for deployment {deployment_id}. Check Railway before retrying.")


def deploy(token, project_id, environment_id, service_id, image, origin):
    scope = graphql(token, "projectToken", PROJECT_SCOPE)
    if scope.get("projectId") != project_id or scope.get("environmentId") != environment_id:
        raise DeployError("The Railway project token does not match the configured project and environment.")

    target = {"serviceId": service_id, "environmentId": environment_id}
    service = graphql(token, "serviceInstance", SERVICE_SETTINGS, **target)
    source = service.get("source") or {}
    current_image = (source.get("image") or "").split("@", 1)[0].split(":", 1)[0]
    if source.get("repo") or current_image != image.rsplit(":", 1)[0]:
        raise DeployError("The Railway service must already use a Docker image from this repo's GHCR package.")
    if service.get("healthcheckPath") != "/healthz" or service.get("startCommand"):
        raise DeployError("Set Railway's healthcheck to /healthz and leave its Start Command blank.")
    if service.get("numReplicas") not in (None, 1):
        raise DeployError("CerviAI requires one Railway replica; recent reports are local to that replica.")
    if not any(origin == f"https://{domain['domain']}" for domain in service["domains"]["serviceDomains"]):
        raise DeployError("CERVIAI_APP_URL must be the generated HTTPS domain of this Railway service.")

    updated = graphql(token, "serviceInstanceUpdate", UPDATE_IMAGE, **target, input={"source": {"image": image}})
    if updated is not True:
        raise DeployError("Railway did not confirm the image source update.")
    # A regular redeploy can reuse the old image reference. Create a new deployment.
    deployment_id = graphql(token, "serviceInstanceDeployV2", DEPLOY_IMAGE, **target)
    if not isinstance(deployment_id, str) or not ID_PATTERN.fullmatch(deployment_id):
        raise DeployError("Railway did not return a valid deployment ID. Check Railway before retrying.")
    print(f"Started Railway deployment {deployment_id} for {image}.")
    wait_for_deployment(token, deployment_id)
    print(f"Railway deployment {deployment_id}: SUCCESS")
    return deployment_id


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise DeployError(f"Missing {name}. See the Railway deployment settings in instructions.txt.")
    return value


def railway_id(name):
    value = required(name).lower()
    if not ID_PATTERN.fullmatch(value):
        raise DeployError(f"{name} must be a Railway ID (UUID), not a name or URL.")
    return value


def published_image():
    image = required("CERVIAI_IMAGE_TAG")
    prefix = f"ghcr.io/{required('GITHUB_REPOSITORY').lower()}:{required('GITHUB_SHA')}-{required('GITHUB_RUN_ID')}-"
    # A retry of only the deploy job can reuse the publish output from an earlier attempt.
    if not re.fullmatch(re.escape(prefix) + r"[1-9][0-9]*", image):
        raise DeployError("Expected the unique GHCR tag published by this GitHub Actions run.")
    return image


def main():
    try:
        origin = railway_origin(required("CERVIAI_APP_URL"))
        deployment_id = deploy(
            required("RAILWAY_PROJECT_TOKEN"), railway_id("RAILWAY_PROJECT_ID"),
            railway_id("RAILWAY_ENVIRONMENT_ID"), railway_id("RAILWAY_SERVICE_ID"), published_image(), origin,
        )
    except (DeployError, ValueError) as error:
        sys.exit(f"Railway deploy failed: {error}")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        print(f"id={deployment_id}\nurl={origin}", file=output)


if __name__ == "__main__":
    main()
