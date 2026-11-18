#!/usr/bin/env bash
# Run in Google Cloud Shell after completing Steps 1–3 of instructions.txt.
set +x
set -euo pipefail

read -r -p 'Dedicated Google Cloud Project ID: ' project
read -r -p 'Full ghcr.io Image from the GitHub Summary: ' image
if [[ ! "$project" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ || ! "$image" =~ ^ghcr\.io/[a-z0-9_.-]+/[a-z0-9_.-]+@sha256:[0-9a-f]{64}$ ]]; then
  echo 'Check the project ID and copy the entire Image line (including @sha256:...). Nothing was created.' >&2
  exit 1
fi
read -r -s -p 'Paste the app-only Neon URL from prepare_neon.py (hidden): ' db_url
printf '\n'
if [[ "$db_url" != postgresql://cerviai_app:* || ! "$db_url" =~ [\?\&]sslmode=verify-full(\&|$) ]]; then
  echo 'Use the app-only pooled URL from prepare_neon.py, not the Neon owner URL.' >&2
  exit 1
fi
read -r -s -p 'Paste your new invitation password (hidden), then press Enter: ' demo_password
printf '\n'
if (( ${#demo_password} < 24 )); then
  echo 'Use a random invitation password at least 24 characters long.' >&2
  exit 1
fi
read -r -s -p 'Optional Groq API key (hidden; press Enter to leave off): ' groq_key
printf '\n'

printf 'This will create/update cerviai-demo in %s (us-central1); Cloud Run charges can apply.\n' "$project"
read -r -p 'Type YES to continue: ' confirmation
if [[ "$confirmation" != YES ]]; then
  echo 'Cancelled. No resources were changed.'
  exit 0
fi

cloud() { gcloud --project="$project" "$@"; }
cloud services enable run.googleapis.com secretmanager.googleapis.com iam.googleapis.com
runtime="cerviai-runtime@${project}.iam.gserviceaccount.com"
if ! cloud iam service-accounts describe "$runtime" >/dev/null 2>&1; then
  cloud iam service-accounts create cerviai-runtime --display-name='CerviAI demo runtime'
fi

# Bind numeric versions so a revision always starts with the same settings.
secret_binding() {
  local version=${2##*/}
  if [[ ! "$version" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the version of $1." >&2
    return 1
  fi
  printf '%s:%s' "$1" "$version"
}

store_secret() {
  local resource
  if ! cloud secrets describe "$1" >/dev/null 2>&1; then
    cloud secrets create "$1" --replication-policy=automatic --quiet >/dev/null || return 1
  fi
  resource=$(cloud secrets versions add "$1" --data-file=- --format='value(name)' --quiet) || return 1
  secret_binding "$1" "$resource"
}

db_binding=$(printf '%s' "$db_url" | store_secret cerviai-db-url)
demo_binding=$(printf '%s' "$demo_password" | store_secret cerviai-demo-password)
if cloud secrets describe cerviai-session-key >/dev/null 2>&1; then
  resource=$(cloud secrets versions describe latest --secret=cerviai-session-key --format='value(name)')
  session_binding=$(secret_binding cerviai-session-key "$resource")
else
  session_binding=$(python3 -c 'import secrets; print(secrets.token_hex(32), end="")' | store_secret cerviai-session-key)
fi
bindings="DATABASE_URL=$db_binding,SECRET_KEY=$session_binding,DEMO_PASSWORD=$demo_binding"
names=(cerviai-db-url cerviai-session-key cerviai-demo-password)
if [[ -n "$groq_key" ]]; then
  groq_binding=$(printf '%s' "$groq_key" | store_secret cerviai-groq-key)
  names+=(cerviai-groq-key)
  bindings+=",GROQ_API_KEY=$groq_binding"
fi
unset db_url demo_password groq_key

for name in "${names[@]}"; do
  cloud secrets add-iam-policy-binding "$name" \
    --member="serviceAccount:$runtime" --role=roles/secretmanager.secretAccessor --quiet >/dev/null
done
cloud run deploy cerviai-demo --image="$image" --region=us-central1 \
  --service-account="$runtime" --set-secrets="$bindings" --no-invoker-iam-check \
  --execution-environment=gen2 --port=8080 --cpu=2 --memory=4Gi \
  --cpu-throttling --no-cpu-boost --min=0 --max=1 --concurrency=1 --timeout=300 \
  --startup-probe=httpGet.path=/healthz,failureThreshold=6,periodSeconds=10,timeoutSeconds=5 \
  --deploy-health-check --quiet
printf 'Service URL: '
cloud run services describe cerviai-demo --region=us-central1 --format='value(status.url)'
