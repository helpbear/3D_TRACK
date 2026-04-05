# Surgical_Multi-Task_AI_Assistant

Web-based scaffold for a surgical multi-task AI assistant focused on endoscopic scene understanding.

## Structure

- `backend/`: FastAPI service for health checks, surgical scene analysis, and model-facing APIs
- `frontend/`: Vite + React client for the surgical operator dashboard
- `main.py`: local backend entrypoint

## Backend

Create or activate the `3dtrack` environment, then install backend dependencies:

```powershell
conda activate 3dtrack
python -m pip install -r backend/requirements.txt
python main.py
```

The backend runs on `http://127.0.0.1:8000`.

### Surgical scene assistant endpoint

The main project route now targets the surgical multi-task requirement directly:

```powershell
curl http://127.0.0.1:8000/api/v1/tasks
curl -X POST http://127.0.0.1:8000/api/v1/assistant/analyze-frame `
  -F "task_id=activity-recognition" `
  -F "procedure=endoscopic-submucosal-dissection" `
  -F "note=possible smoke near the lesion edge" `
  -F "frame=@C:\path\to\endoscopy_frame.png"
```

The response returns activity scores, visibility and risk triage, recommended overlays, and next-step guidance.

### Experimental TUS-REC route

After cloning `tus-rec-challenge_baseline` into the project root, the backend can load its bundled
checkpoint and expose a real inference route:

```powershell
curl http://127.0.0.1:8000/api/v1/tus-rec/status
curl -X POST http://127.0.0.1:8000/api/v1/tus-rec/predict `
  -F "frame_0=@C:\path\to\frame0.png" `
  -F "frame_1=@C:\path\to\frame1.png"
```

The response returns the raw 6-parameter prediction in the repository's `ZYX + translation`
format and the derived 4x4 transform matrix. This route is kept as a separate experiment and is
not the main path for the surgical assistant project.

## Frontend

Install the UI dependencies and start the dev server:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs on `http://127.0.0.1:5173` and proxies `/api` to the backend. The main page now
supports uploading an endoscopic frame and reviewing multi-task scene analysis outputs.

## Next steps

- Replace the heuristic frame triage in `backend/app/services/surgical_assistant.py` with a real endoscopic multitask model
- Add video upload, clip sampling, and temporal smoothing
- Persist sessions, uploaded frames, and operator decisions
