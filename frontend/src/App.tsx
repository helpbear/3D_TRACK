import {
  FormEvent,
  startTransition,
  useEffect,
  useState,
} from "react";

type HealthResponse = {
  status: "ok";
  service: string;
};

type TaskCard = {
  id: string;
  name: string;
  summary: string;
  category: "classification" | "segmentation" | "workflow" | "safety";
  outputs: string[];
  status: "planned" | "prototype" | "ready";
};

type TaskCatalogResponse = {
  items: TaskCard[];
};

type ActivityScore = {
  label: "marking" | "injection" | "dissection" | "idle";
  score: number;
};

type OverlayTarget = {
  id: string;
  label: string;
  kind: "region" | "instrument" | "anatomy";
  priority: "primary" | "supporting";
};

type RiskFlag = {
  code: string;
  severity: "info" | "warning" | "critical";
  message: string;
};

type AnalysisResponse = {
  task_id: string;
  procedure: string;
  frame_name: string | null;
  scene_summary: string;
  primary_activity: "marking" | "injection" | "dissection" | "idle";
  activity_scores: ActivityScore[];
  visibility_score: number;
  risk_flags: RiskFlag[];
  recommended_overlays: OverlayTarget[];
  instrument_hints: string[];
  safe_to_continue: boolean;
  confidence: number;
  findings: string[];
  next_action: string;
};

const emptyResult: AnalysisResponse | null = null;

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [tasks, setTasks] = useState<TaskCard[]>([]);
  const [taskId, setTaskId] = useState("activity-recognition");
  const [procedure, setProcedure] = useState("endoscopic-submucosal-dissection");
  const [note, setNote] = useState("");
  const [frameFile, setFrameFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(emptyResult);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    if (!frameFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(frameFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [frameFile]);

  const selectedTask = tasks.find((task) => task.id === taskId) ?? null;

  async function bootstrap() {
    try {
      const [healthRes, tasksRes] = await Promise.all([
        fetch("/api/v1/health"),
        fetch("/api/v1/tasks"),
      ]);

      if (!healthRes.ok || !tasksRes.ok) {
        throw new Error("Failed to load backend bootstrap data.");
      }

      const healthData = (await healthRes.json()) as HealthResponse;
      const taskData = (await tasksRes.json()) as TaskCatalogResponse;

      startTransition(() => {
        setHealth(healthData);
        setTasks(taskData.items);
        if (taskData.items.length > 0) {
          setTaskId(taskData.items[0].id);
        }
      });
    } catch (bootstrapError) {
      setError(
        bootstrapError instanceof Error
          ? bootstrapError.message
          : "Unknown bootstrap error."
      );
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!frameFile) {
      setError("Select an endoscopic frame before running scene analysis.");
      return;
    }

    setBusy(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.set("task_id", taskId);
      formData.set("procedure", procedure);
      formData.set("note", note);
      formData.set("frame", frameFile);

      const response = await fetch("/api/v1/assistant/analyze-frame", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis request failed.");
      }

      const data = (await response.json()) as AnalysisResponse;
      startTransition(() => {
        setResult(data);
      });
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Unknown submission error."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <section className="hero panel">
        <div>
          <p className="eyebrow">3D Track</p>
          <h1>Surgical Multi-Task AI Assistant</h1>
          <p className="lede">
            A browser-based control surface for endoscopic scene understanding,
            workflow support, overlay planning, and surgical safety monitoring.
          </p>
        </div>
        <div className="status-card">
          <span className={`dot ${health?.status === "ok" ? "live" : ""}`} />
          <div>
            <p className="status-label">Backend status</p>
            <strong>{health ? `${health.status} / ${health.service}` : "loading"}</strong>
          </div>
        </div>
      </section>

      <section className="grid">
        <article className="panel">
          <div className="section-heading">
            <h2>Task catalog</h2>
            <span>{tasks.length} modules</span>
          </div>
          <div className="task-list">
            {tasks.map((task) => (
              <div key={task.id} className="task-card">
                <div className="task-row">
                  <strong>{task.name}</strong>
                  <span className={`badge badge-${task.status}`}>{task.status}</span>
                </div>
                <p className="task-meta">{task.category}</p>
                <p>{task.summary}</p>
                <p className="task-outputs">{task.outputs.join(" • ")}</p>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="section-heading">
            <h2>Frame triage</h2>
            <span>{selectedTask?.category ?? "assistant"}</span>
          </div>
          <form className="analysis-form" onSubmit={handleSubmit}>
            <label>
              <span>Workflow module</span>
              <select value={taskId} onChange={(event) => setTaskId(event.target.value)}>
                {tasks.map((task) => (
                  <option key={task.id} value={task.id}>
                    {task.name}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Procedure</span>
              <select value={procedure} onChange={(event) => setProcedure(event.target.value)}>
                <option value="endoscopic-submucosal-dissection">
                  Endoscopic Submucosal Dissection
                </option>
                <option value="endoscopic-mucosal-resection">
                  Endoscopic Mucosal Resection
                </option>
                <option value="lesion-screening">Lesion Screening</option>
              </select>
            </label>

            <label>
              <span>Endoscopic frame</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(event) => setFrameFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <label>
              <span>Operator note</span>
              <textarea
                rows={6}
                value={note}
                onChange={(event) => setNote(event.target.value)}
                placeholder="Example: smoke around the lesion edge, likely entering circumferential incision, possible minor bleeding."
              />
            </label>

            <button type="submit" disabled={busy}>
              {busy ? "Analyzing frame..." : "Analyze frame"}
            </button>
          </form>
        </article>
      </section>

      <section className="grid secondary-grid">
        <article className="panel preview-panel">
          <div className="section-heading">
            <h2>Frame preview</h2>
            <span>{frameFile?.name ?? "no file selected"}</span>
          </div>
          {previewUrl ? (
            <img className="frame-preview" src={previewUrl} alt="Endoscopic frame preview" />
          ) : (
            <p className="placeholder">
              Upload an endoscopic frame to drive activity recognition and overlay planning.
            </p>
          )}
        </article>

        <article className="panel">
          <div className="section-heading">
            <h2>Scene focus</h2>
            <span>{result ? result.primary_activity : "waiting"}</span>
          </div>
          {result ? (
            <div className="summary-stack">
              <p className="scene-summary">{result.scene_summary}</p>
              <div className="chip-row">
                {result.instrument_hints.map((hint) => (
                  <span key={hint} className="chip">
                    {hint}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <p className="placeholder">
              The frame analysis will summarize the current activity, visibility, and likely overlays.
            </p>
          )}
        </article>
      </section>

      <section className="panel result-panel">
        <div className="section-heading">
          <h2>Latest response</h2>
          <span>{result ? result.frame_name ?? result.task_id : "waiting"}</span>
        </div>

        {error ? <p className="error-banner">{error}</p> : null}

        {result ? (
          <div className="result-grid">
            <div className="metric">
              <span>Confidence</span>
              <strong>{Math.round(result.confidence * 100)}%</strong>
            </div>
            <div className="metric">
              <span>Visibility</span>
              <strong>{Math.round(result.visibility_score * 100)}%</strong>
            </div>
            <div className="metric">
              <span>Continue</span>
              <strong>{result.safe_to_continue ? "yes" : "hold"}</strong>
            </div>
            <div className="metric">
              <span>Primary activity</span>
              <strong>{result.primary_activity}</strong>
            </div>
            <div className="metric wide">
              <span>Next action</span>
              <strong>{result.next_action}</strong>
            </div>
            <div className="wide findings">
              <span>Activity scores</span>
              <ul className="score-list">
                {result.activity_scores.map((item) => (
                  <li key={item.label}>
                    <span>{item.label}</span>
                    <strong>{Math.round(item.score * 100)}%</strong>
                  </li>
                ))}
              </ul>
            </div>
            <div className="wide findings">
              <span>Recommended overlays</span>
              <ul>
                {result.recommended_overlays.map((overlay) => (
                  <li key={overlay.id}>
                    {overlay.label} ({overlay.kind}, {overlay.priority})
                  </li>
                ))}
              </ul>
            </div>
            <div className="wide findings">
              <span>Risk flags</span>
              <ul>
                {result.risk_flags.map((flag) => (
                  <li key={flag.code}>
                    [{flag.severity}] {flag.message}
                  </li>
                ))}
              </ul>
            </div>
            <div className="wide findings">
              <span>Findings</span>
              <ul>
                {result.findings.map((finding) => (
                  <li key={finding}>{finding}</li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <p className="placeholder">
            Upload a surgical frame to inspect the new project-oriented response contract before the real multitask model is attached.
          </p>
        )}
      </section>
    </main>
  );
}
