#!/usr/bin/env bash
# Run in Google Cloud Shell after the initial service passes Step 5.
set -euo pipefail

read -r -p 'Google Cloud Project ID used in Step 4: ' project
read -r -p 'Repository ID from the GitHub Summary (digits): ' repo_id
read -r -p 'Owner ID from the GitHub Summary (digits): ' owner_id
read -r -p 'Branch from the GitHub Summary (usually main): ' branch
if [[ ! "$project" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ || ! "$repo_id" =~ ^[0-9]+$ || ! "$owner_id" =~ ^[0-9]+$ || ! "$branch" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo 'Check the project ID, the two numeric IDs and the branch name.' >&2
  exit 1
fi

cloud() { gcloud --project="$project" "$@"; }
cloud run services describe cerviai-demo --region=us-central1 >/dev/null
cloud services enable iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com
project_number=$(cloud projects describe "$project" --format='value(projectNumber)')
if [[ ! "$project_number" =~ ^[0-9]+$ ]]; then
  echo 'Could not find the Google Cloud project number.' >&2
  exit 1
fi
pool=cerviai-github
provider=github
deployer="cerviai-deployer@${project}.iam.gserviceaccount.com"
runtime="cerviai-runtime@${project}.iam.gserviceaccount.com"

if ! cloud iam service-accounts describe "$deployer" >/dev/null 2>&1; then
  cloud iam service-accounts create cerviai-deployer --display-name='CerviAI GitHub deployer'
fi
if ! cloud iam workload-identity-pools describe "$pool" --location=global >/dev/null 2>&1; then
  cloud iam workload-identity-pools create "$pool" --location=global --display-name='CerviAI GitHub Actions'
fi
mapping='google.subject=assertion.sub,attribute.repository_id=assertion.repository_id'
restriction="assertion.repository_id=='${repo_id}' && assertion.repository_owner_id=='${owner_id}' && assertion.ref=='refs/heads/${branch}' && assertion.event_name=='workflow_dispatch' && assertion.workflow_ref.endsWith('/.github/workflows/deploy-cloud-run.yml@refs/heads/${branch}')"
args=(--location=global --workload-identity-pool="$pool" --attribute-mapping="$mapping" --attribute-condition="$restriction" --issuer-uri=https://token.actions.githubusercontent.com)
if cloud iam workload-identity-pools providers describe "$provider" --location=global --workload-identity-pool="$pool" >/dev/null 2>&1; then
  cloud iam workload-identity-pools providers update-oidc "$provider" "${args[@]}"
else
  cloud iam workload-identity-pools providers create-oidc "$provider" "${args[@]}"
fi
pool_path="projects/$project_number/locations/global/workloadIdentityPools/$pool"
cloud iam service-accounts add-iam-policy-binding "$deployer" \
  --member="principalSet://iam.googleapis.com/$pool_path/attribute.repository_id/$repo_id" \
  --role=roles/iam.workloadIdentityUser --quiet
cloud iam service-accounts add-iam-policy-binding "$runtime" \
  --member="serviceAccount:$deployer" --role=roles/iam.serviceAccountUser --quiet
cloud run services add-iam-policy-binding cerviai-demo --region=us-central1 \
  --member="serviceAccount:$deployer" --role=roles/run.developer --quiet
printf '\nCopy these three values to GitHub repository Variables:\n'
printf 'GCP_PROJECT_ID=%s\nGCP_WIF_PROVIDER=%s/providers/%s\nGCP_DEPLOYER_SA=%s\n' \
  "$project" "$pool_path" "$provider" "$deployer"
