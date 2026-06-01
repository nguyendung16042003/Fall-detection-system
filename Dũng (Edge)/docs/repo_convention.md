# Repository Convention — Fall Detection Edge–Cloud System

**Owner:** Anh Dũng — Edge / Architecture  
**Version:** Week 1 draft

---

## 1. Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable version, only merge after tested |
| `dev` | Integration branch for all members |
| `feature/<name>` | Feature branches |
| `fix/<name>` | Bug fix branches |
| `docs/<name>` | Documentation-only changes |

Examples:

```text
feature/edge-mqtt-publisher
feature/cloud-event-api
feature/app-alert-screen
feature/model-yolov8-cls-v1
fix/app-empty-history
```

---

## 2. Commit message format

Format:

```text
type(scope): short description
```

Allowed `type`:

| Type | Meaning |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `config` | Config file change |
| `refactor` | Code restructuring |
| `test` | Test code |
| `chore` | Other maintenance |

Allowed `scope`:

| Scope | Meaning |
|---|---|
| `edge` | Jetson, DeepStream, MQTT publisher |
| `model` | AI model, training, TensorRT export |
| `cloud` | FastAPI, MQTT consumer, storage, notification |
| `app` | Flutter mobile/desktop app |
| `docs` | Project docs |
| `deploy` | Docker, deployment, scripts |

Examples:

```text
docs(edge): add MQTT payload schema
feat(cloud): add event history API
feat(app): implement alert detail screen
config(deploy): add postgres and mqtt broker services
fix(edge): prevent duplicate fall event publishing
```

---

## 3. Repository folder structure

```text
fall-detection-system/
  edge/
    deepstream/
      configs/
      custom_parser/
      probes/
    scripts/
    README.md

  model/
    datasets/
    training/
    export/
    evaluation/
    README.md

  cloud/
    backend/
      app/
        routers/
        models/
        schemas/
        services/
        workers/
      Dockerfile
    mosquitto/
      config/
    README.md

  app/
    flutter_app/
    README.md

  docs/
    architecture.md
    mqtt_schema.json
    api_contract.md
    repo_convention.md

  docker-compose.skeleton.yml
  README.md
```

---

## 4. Pull request rule

Before merge into `dev`:

- Code runs locally or has clear note if not testable yet.
- No secret key, password, API key committed.
- API/MQTT changes must update `docs/api_contract.md` or `docs/mqtt_schema.json`.
- Each PR should be reviewed by at least one related member.

---

## 5. Environment and secret rule

Never commit real secrets.

Use `.env.example`:

```env
DATABASE_URL=postgresql://fall_user:fall_password@postgres:5432/fall_detection
MQTT_HOST=mqtt-broker
MQTT_PORT=1883
MINIO_ENDPOINT=http://minio:9000
FIREBASE_CREDENTIALS_PATH=/run/secrets/firebase-admin.json
```

Real `.env` must be ignored by Git.

Add to `.gitignore`:

```gitignore
.env
*.engine
*.onnx
*.pt
*.mp4
*.jpg
*.png
__pycache__/
.vscode/
.idea/
```

---

## 6. Week 1 required files

Anh Dũng should deliver these files to unblock the team:

```text
docs/architecture.md
docs/mqtt_schema.json
docs/api_contract.md
docs/repo_convention.md
docker-compose.skeleton.yml
```
