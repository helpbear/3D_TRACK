from io import BytesIO

import numpy as np
from PIL import Image

from backend.app.schemas.assistant import (
    ActivityScore,
    AnalysisRequest,
    AnalysisResponse,
    OverlayTarget,
    RiskFlag,
    TaskCard,
)
from backend.app.services.endoarss_runtime import get_endoarss_runtime

TASKS = [
    TaskCard(
        id="activity-recognition",
        name="Activity Recognition",
        summary="Classify the current operative action into marking, injection, dissection, or idle.",
        category="classification",
        outputs=["activity label", "activity score"],
        status="prototype",
    ),
    TaskCard(
        id="region-and-tool-segmentation",
        name="Region And Tool Segmentation",
        summary="Highlight the knife tip, lesion boundary, marking area, and dissection plane.",
        category="segmentation",
        outputs=["recommended overlays", "instrument cues"],
        status="prototype",
    ),
    TaskCard(
        id="workflow-phase-estimation",
        name="Workflow Phase Estimation",
        summary="Track lesion exposure, circumferential incision, and submucosal dissection progress.",
        category="workflow",
        outputs=["phase summary", "next action"],
        status="planned",
    ),
    TaskCard(
        id="visibility-and-risk-monitoring",
        name="Visibility And Risk Monitoring",
        summary="Flag smoke, glare, possible bleeding, and unstable visibility before the next action.",
        category="safety",
        outputs=["visibility score", "risk flags"],
        status="prototype",
    ),
]


def list_tasks() -> list[TaskCard]:
    return TASKS


def analyze_request(payload: AnalysisRequest) -> AnalysisResponse:
    profile = _build_scene_profile(
        task_id=payload.task_id,
        procedure=payload.procedure,
        note=payload.note,
        frame_name=None,
        image_rgb=None,
        model_output=None,
    )
    return AnalysisResponse(**profile)


def analyze_frame(
    task_id: str,
    procedure: str,
    note: str,
    frame_name: str,
    frame_bytes: bytes,
) -> AnalysisResponse:
    if not frame_bytes:
        raise ValueError("Uploaded frame is empty.")

    image_rgb = Image.open(BytesIO(frame_bytes)).convert("RGB")
    model_output = get_endoarss_runtime().predict(image_rgb)
    profile = _build_scene_profile(
        task_id=task_id,
        procedure=procedure,
        note=note,
        frame_name=frame_name,
        image_rgb=image_rgb,
        model_output=model_output,
    )
    return AnalysisResponse(**profile)


def _build_scene_profile(
    task_id: str,
    procedure: str,
    note: str,
    frame_name: str | None,
    image_rgb: Image.Image | None,
    model_output: dict | None,
) -> dict:
    note_lower = note.lower()
    stats = _compute_frame_stats(image_rgb)
    activity_scores = _infer_activity_scores(stats, note_lower, model_output)
    primary_activity = max(activity_scores, key=lambda item: item.score).label
    risk_flags = _infer_risk_flags(stats, note_lower, model_output, activity_scores)
    visibility_score = _compute_visibility(stats, risk_flags, model_output)
    confidence = _compute_confidence(
        activity_scores,
        visibility_score,
        note_lower,
        model_output,
    )
    overlays = _recommended_overlays(primary_activity, task_id, model_output)
    instrument_hints = _instrument_hints(primary_activity, note_lower, model_output)
    safe_to_continue = not any(flag.severity == "critical" for flag in risk_flags)
    findings = _findings(stats, note_lower, primary_activity, model_output)
    next_action = _next_action(primary_activity, risk_flags)

    if image_rgb is None:
        scene_summary = (
            "Text-only triage mode. Attach an endoscopic frame to drive scene-aware "
            "activity recognition and overlay selection."
        )
    else:
        scene_summary = (
            f"The uploaded endoscopic frame is most consistent with `{primary_activity}` activity "
            f"for the `{procedure}` workflow, with visibility scored at {round(visibility_score * 100)}%."
        )
        if model_output is not None:
            scene_summary += " The result is driven by the EndoARSS MTLESD multitask checkpoint."

    return {
        "task_id": task_id,
        "procedure": procedure,
        "frame_name": frame_name,
        "scene_summary": scene_summary,
        "primary_activity": primary_activity,
        "activity_scores": activity_scores,
        "visibility_score": visibility_score,
        "risk_flags": risk_flags,
        "recommended_overlays": overlays,
        "instrument_hints": instrument_hints,
        "safe_to_continue": safe_to_continue,
        "confidence": confidence,
        "findings": findings,
        "next_action": next_action,
    }


def _compute_frame_stats(image_rgb: Image.Image | None) -> dict[str, float]:
    if image_rgb is None:
        return {
            "brightness": 0.52,
            "contrast": 0.24,
            "red_dominance": 0.36,
            "green_dominance": 0.31,
            "highlight_ratio": 0.03,
            "dark_ratio": 0.08,
            "warm_region_ratio": 0.22,
        }

    pixels = np.asarray(image_rgb, dtype=np.float32)
    gray = pixels.mean(axis=2)
    channel_means = pixels.mean(axis=(0, 1)) + 1e-6
    pixel_mean = float(channel_means.mean())

    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "red_dominance": float(channel_means[0] / pixel_mean),
        "green_dominance": float(channel_means[1] / pixel_mean),
        "highlight_ratio": float(np.mean(pixels.max(axis=2) > 245)),
        "dark_ratio": float(np.mean(gray < 40)),
        "warm_region_ratio": float(
            np.mean(
                (pixels[:, :, 0] > pixels[:, :, 1] * 1.15)
                & (pixels[:, :, 0] > pixels[:, :, 2] * 1.15)
                & (pixels[:, :, 0] > 110)
            )
        ),
    }


def _infer_activity_scores(
    stats: dict[str, float],
    note_lower: str,
    model_output: dict | None,
) -> list[ActivityScore]:
    if model_output is not None:
        model_scores = dict(model_output["classification"])
        idle_raw = max(
            0.0,
            0.45 - max(model_scores.values()),
        ) * 1.6
        raw_scores = {
            "marking": model_scores.get("marking", 0.0),
            "injection": model_scores.get("injection", 0.0),
            "dissection": model_scores.get("dissection", 0.0),
            "idle": idle_raw,
        }
        total = sum(raw_scores.values()) or 1.0
        return [
            ActivityScore(label=label, score=round(value / total, 4))
            for label, value in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        ]

    keyword_boosts = {
        "marking": 0.18 if "mark" in note_lower else 0.0,
        "injection": 0.18 if "inject" in note_lower or "lift" in note_lower else 0.0,
        "dissection": 0.2 if "dissect" in note_lower or "knife" in note_lower else 0.0,
        "idle": 0.14 if "idle" in note_lower or "waiting" in note_lower else 0.0,
    }

    raw_scores = {
        "marking": 0.22
        + stats["red_dominance"] * 0.18
        + stats["warm_region_ratio"] * 0.32
        + keyword_boosts["marking"],
        "injection": 0.2
        + stats["green_dominance"] * 0.16
        + (1.0 - abs(stats["brightness"] - 0.55)) * 0.18
        + keyword_boosts["injection"],
        "dissection": 0.22
        + stats["contrast"] * 0.5
        + stats["highlight_ratio"] * 0.35
        + keyword_boosts["dissection"],
        "idle": 0.18
        + (1.0 - stats["contrast"]) * 0.22
        + (1.0 - stats["highlight_ratio"]) * 0.12
        + keyword_boosts["idle"],
    }

    total = sum(raw_scores.values())
    return [
        ActivityScore(label=label, score=round(value / total, 4))
        for label, value in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _infer_risk_flags(
    stats: dict[str, float],
    note_lower: str,
    model_output: dict | None,
    activity_scores: list[ActivityScore],
) -> list[RiskFlag]:
    risk_flags: list[RiskFlag] = []
    top_activity_score = activity_scores[0].score

    if "bleed" in note_lower or stats["warm_region_ratio"] > 0.42:
        risk_flags.append(
            RiskFlag(
                code="bleeding-suspected",
                severity="critical",
                message="Warm tissue dominance suggests possible bleeding or active mucosal trauma.",
            )
        )
    if "smoke" in note_lower or (stats["contrast"] < 0.12 and stats["brightness"] > 0.35):
        risk_flags.append(
            RiskFlag(
                code="visibility-degraded",
                severity="warning",
                message="Low-contrast appearance suggests smoke, blur, or lens contamination.",
            )
        )
    if stats["highlight_ratio"] > 0.1:
        risk_flags.append(
            RiskFlag(
                code="specular-glare",
                severity="warning",
                message="Strong specular reflections may hide the knife edge or lesion boundary.",
            )
        )
    if stats["dark_ratio"] > 0.18:
        risk_flags.append(
            RiskFlag(
                code="underexposed",
                severity="info",
                message="A large dark region reduces confidence around the distal field of view.",
            )
        )
    if model_output is not None and top_activity_score < 0.55:
        risk_flags.append(
            RiskFlag(
                code="low-model-certainty",
                severity="warning",
                message="EndoARSS is uncertain about the current activity, so downstream guidance should be reviewed manually.",
            )
        )
    if model_output is not None and model_output["region_coverage"].get(0, 0.0) > 0.92:
        risk_flags.append(
            RiskFlag(
                code="weak-foreground-support",
                severity="info",
                message="The segmentation head assigned most pixels to background, which usually means weak structural support in this frame.",
            )
        )
    if not risk_flags:
        risk_flags.append(
            RiskFlag(
                code="stable-visibility",
                severity="info",
                message="No dominant visibility or bleeding risk was detected in this frame.",
            )
        )

    return risk_flags


def _compute_visibility(
    stats: dict[str, float],
    risk_flags: list[RiskFlag],
    model_output: dict | None,
) -> float:
    visibility = 0.82
    visibility -= stats["highlight_ratio"] * 0.7
    visibility -= max(0.0, 0.16 - stats["contrast"]) * 1.6
    visibility -= stats["dark_ratio"] * 0.25
    visibility -= 0.12 * sum(flag.severity == "critical" for flag in risk_flags)
    if model_output is not None:
        visibility = visibility * 0.55 + model_output["segmentation_confidence"] * 0.45
    return round(float(min(max(visibility, 0.05), 0.99)), 4)


def _compute_confidence(
    activity_scores: list[ActivityScore],
    visibility_score: float,
    note_lower: str,
    model_output: dict | None,
) -> float:
    top_score = activity_scores[0].score
    second_score = activity_scores[1].score
    margin = top_score - second_score
    confidence = 0.48 + margin * 0.85 + visibility_score * 0.18
    if model_output is not None:
        confidence = confidence * 0.6 + model_output["segmentation_confidence"] * 0.4
    if note_lower:
        confidence += 0.04
    return round(float(min(max(confidence, 0.05), 0.98)), 4)


def _recommended_overlays(
    primary_activity: str,
    task_id: str,
    model_output: dict | None,
) -> list[OverlayTarget]:
    overlays_by_activity = {
        "marking": [
            OverlayTarget(id="marking-area", label="Marking area", kind="region", priority="primary"),
            OverlayTarget(id="knife-tip", label="Electric knife tip", kind="instrument", priority="supporting"),
            OverlayTarget(id="lesion-boundary", label="Lesion boundary", kind="anatomy", priority="supporting"),
        ],
        "injection": [
            OverlayTarget(id="injection-point", label="Injection point", kind="region", priority="primary"),
            OverlayTarget(id="needle-path", label="Needle path", kind="instrument", priority="supporting"),
            OverlayTarget(id="submucosal-lift", label="Submucosal lift", kind="region", priority="supporting"),
        ],
        "dissection": [
            OverlayTarget(id="dissection-plane", label="Dissection plane", kind="region", priority="primary"),
            OverlayTarget(id="mucosal-flap", label="Mucosal flap", kind="anatomy", priority="supporting"),
            OverlayTarget(id="knife-tip", label="Electric knife tip", kind="instrument", priority="supporting"),
        ],
        "idle": [
            OverlayTarget(id="lumen-context", label="Lumen context", kind="anatomy", priority="primary"),
            OverlayTarget(id="lesion-boundary", label="Lesion boundary", kind="anatomy", priority="supporting"),
        ],
    }
    overlays = overlays_by_activity[primary_activity]
    if model_output is not None and model_output["dominant_region_ids"]:
        overlays = overlays + [
            OverlayTarget(
                id=f"region-{region_id}",
                label=f"Region {region_id}",
                kind="region",
                priority="supporting",
            )
            for region_id in model_output["dominant_region_ids"][:1]
        ]
    if task_id == "visibility-and-risk-monitoring":
        return overlays[:2]
    return overlays


def _instrument_hints(
    primary_activity: str,
    note_lower: str,
    model_output: dict | None,
) -> list[str]:
    hints = {
        "marking": ["electrocautery marker", "lesion edge reference"],
        "injection": ["injection needle", "submucosal lifting pocket"],
        "dissection": ["electric knife", "traction-assisted mucosal flap"],
        "idle": ["scope tip", "field-of-view stabilization"],
    }[primary_activity][:]

    if "clip" in note_lower:
        hints.append("closure clip")
    if "bleed" in note_lower:
        hints.append("hemostatic tool review")
    if model_output is not None and model_output["dominant_region_ids"]:
        hints.append(f"segmentation region {model_output['dominant_region_ids'][0]}")
    return hints


def _findings(
    stats: dict[str, float],
    note_lower: str,
    primary_activity: str,
    model_output: dict | None,
) -> list[str]:
    findings = [
        f"Primary activity estimate is `{primary_activity}`.",
        f"Estimated brightness {round(stats['brightness'] * 100)}% and contrast {round(stats['contrast'] * 100)}%.",
        f"Warm tissue coverage is {round(stats['warm_region_ratio'] * 100)}%, useful for bleeding and marking cues.",
    ]
    if model_output is not None:
        dominant_regions = [
            f"region {region_id}: {round(model_output['region_coverage'].get(region_id, 0.0) * 100, 1)}%"
            for region_id in model_output["dominant_region_ids"][:3]
        ]
        findings.append(
            "EndoARSS MTLESD checkpoint provided the activity and segmentation predictions."
        )
        if dominant_regions:
            findings.append(
                "Dominant segmentation coverage: " + ", ".join(dominant_regions) + "."
            )
    if note_lower:
        findings.append("Operator note was fused into the current triage profile.")
    else:
        findings.append("No operator note was provided, so the result is image-driven only.")
    return findings


def _next_action(primary_activity: str, risk_flags: list[RiskFlag]) -> str:
    if any(flag.severity == "critical" for flag in risk_flags):
        return "Pause the next action, review the frame manually, and confirm hemostasis before continuing."
    if primary_activity == "marking":
        return "Keep the lesion boundary overlay visible and prepare the circumferential incision path."
    if primary_activity == "injection":
        return "Confirm the lifting plane and keep the needle path overlay active before the next injection."
    if primary_activity == "dissection":
        return "Prioritize the dissection-plane overlay and track the mucosal flap before advancing the knife."
    return "Hold a stable view, verify visibility, and wait for the next operative action."
