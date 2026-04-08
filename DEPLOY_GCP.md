# Deploy DeepFake Detector on GCP (Production)

This guide deploys the app to **Google Cloud Run** with a production container.

## 1. Prerequisites

- Google Cloud project with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed (optional if using Cloud Build only)
- Optional for AI explanations + web search in app:
  - `GEMINI_API_KEY`
  - optional `GEMINI_MODEL` (default used in code: `gemini-2.5-flash`)

## 2. Configure GCP Project

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Enable required APIs:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 3. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create deepfake-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Container images for deepfake detector"
```

## 4. Build and Push Container

From repository root (where `Dockerfile` exists):

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/deepfake-repo/deepfake-detector:latest
```

## 5. Deploy to Cloud Run

```bash
gcloud run deploy deepfake-detector \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/deepfake-repo/deepfake-detector:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --set-env-vars GEMINI_API_KEY=YOUR_GEMINI_API_KEY,GEMINI_MODEL=gemini-2.5-flash
```

Notes:
- `--timeout 900` helps long media processing requests.
- `--memory 4Gi` and `--cpu 2` are practical starting values for OpenCV/Torch workloads.

## 6. Production Secret Handling (Recommended)

Use Secret Manager instead of plain env vars:

```bash
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY --data-file=-

gcloud run services update deepfake-detector \
  --region us-central1 \
  --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest
```

## 7. Verify Deployment

Get the service URL:

```bash
gcloud run services describe deepfake-detector \
  --region us-central1 \
  --format='value(status.url)'
```

Open the URL and test:
- video upload flow
- audio upload flow (`.wav`, `.mp3`, etc.)
- result page rendering and reasoning cards

## 8. Scaling and Performance Guidance

- Start with Cloud Run autoscaling defaults.
- For heavier workloads, increase memory/CPU and timeout.
- If request latency is too high under load, consider:
  - a background job architecture (Pub/Sub + worker)
  - or GKE/GCE for dedicated long-running inference workers.

## 9. Rollback

List revisions:

```bash
gcloud run revisions list --service deepfake-detector --region us-central1
```

Route traffic to a previous stable revision from Cloud Run console or via CLI.

## 10. Local Docker Test (Optional)

```bash
docker build -t deepfake-detector:local .
docker run --rm -p 8080:8080 -e GEMINI_API_KEY=YOUR_GEMINI_API_KEY deepfake-detector:local
```

Then open `http://127.0.0.1:8080`.
