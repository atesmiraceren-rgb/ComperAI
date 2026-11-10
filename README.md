---
title: Comper AI Analysis API
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Comper AI Analysis API

FastAPI endpoint for the Comper AI HTML client.

## API
- `GET /health`
- `POST /analyze` multipart fields: `video`, optional `player_photo`, `player_name`, `player_number`.

## Important limitation
This starter server performs sampled person detection using a general YOLO model. It does **not** identify a specific footballer from a photo, recognize ball possession, or detect goals. It deliberately returns `clips: []` rather than inventing clip timestamps. A football-specific player re-identification and event model must be added for the requested automatic player clips.

## Deploy
Create a new Hugging Face Space using the Docker SDK, then upload these files. After the Space starts, its public URL is the API base URL; append `/analyze` in the Comper AI HTML page.
