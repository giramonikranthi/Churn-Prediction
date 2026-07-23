from pathlib import Path
import os
import mlflow
import mlflow.pyfunc

from src.config import MLFLOW_TRACKING_URI, MLFLOW_MODEL_URI
from src.logger_utils import logger


def _is_local_tracking_uri(uri: str | None) -> bool:
    """Return True when tracking URI points to localhost/loopback."""
    if not uri:
        return False

    normalized = uri.strip().lower()
    return any(
        token in normalized
        for token in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    )


def _is_streamlit_cloud() -> bool:
    """Best-effort detection for Streamlit Community Cloud runtime."""
    return os.getenv("STREAMLIT_SHARING_MODE", "").lower() == "community"


def _get_requested_model_family(model_name: str) -> str | None:
    """Infer requested model family from configured model name."""
    normalized = (model_name or "").strip().lower()
    if "xgb" in normalized or "xgboost" in normalized:
        return "xgboost"
    if "logistic" in normalized or "random forest" in normalized or "sklearn" in normalized:
        return "sklearn"
    return None


def _is_loaded_model_family_match(model: mlflow.pyfunc.PyFuncModel, requested_family: str | None) -> bool:
    """Validate that loaded pyfunc model contains the requested flavor."""
    if requested_family is None:
        return True

    flavors = getattr(getattr(model, "metadata", None), "flavors", {}) or {}
    if requested_family == "xgboost":
        return "xgboost" in flavors
    if requested_family == "sklearn":
        return "sklearn" in flavors

    return True


def _score_local_candidate(
    mlmodel_path: Path,
    run_id: str,
    requested_family: str | None,
) -> tuple[int, float]:
    """Score local model candidates so best-fit artifacts are attempted first."""
    score = 0

    try:
        mlmodel_text = mlmodel_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        mlmodel_text = ""

    requested_run_id = (run_id or "").strip().lower()
    if requested_run_id and f"run_id: {requested_run_id}" in mlmodel_text:
        score += 300

    if requested_family == "xgboost" and "\n  xgboost:" in mlmodel_text:
        score += 150
    elif requested_family == "sklearn" and "\n  sklearn:" in mlmodel_text:
        score += 150

    if "python_function:" in mlmodel_text:
        score += 10

    return score, mlmodel_path.stat().st_mtime


def _load_local_artifact_model(run_id: str, model_name: str) -> mlflow.pyfunc.PyFuncModel:
    """Try loading the locally saved artifact from the MLruns store."""
    mlruns_root = Path(__file__).resolve().parents[1] / "mlruns"

    raw_candidates = list(mlruns_root.rglob("MLmodel"))
    requested_family = _get_requested_model_family(model_name)

    artifact_candidates = sorted(
        raw_candidates,
        key=lambda p: _score_local_candidate(p, run_id=run_id, requested_family=requested_family),
        reverse=True,
    )
    if not artifact_candidates:
        raise FileNotFoundError("No MLflow artifact directory with an MLmodel file was found.")

    logger.info(
        "Found %s local MLflow artifact candidates (requested family=%s)",
        len(artifact_candidates),
        requested_family or "unknown",
    )

    for mlmodel_path in artifact_candidates:
        artifact_dir = mlmodel_path.parent
        if artifact_dir.exists():
            try:
                logger.info("Attempting to load local artifact model from %s", artifact_dir)
                model = mlflow.pyfunc.load_model(str(artifact_dir))
                if not _is_loaded_model_family_match(model, requested_family):
                    logger.warning(
                        "Skipping local artifact %s because loaded flavor does not match requested family=%s",
                        artifact_dir,
                        requested_family,
                    )
                    continue
                logger.info("Successfully loaded local artifact model from %s", artifact_dir)
                return model
            except Exception as exc:
                logger.warning("Local artifact load failed for %s: %s", artifact_dir, exc)
                continue

    raise FileNotFoundError("Could not load a usable MLflow artifact from the local mlruns folder.")


def _get_latest_model_version(client: mlflow.MlflowClient, model_name: str) -> int | None:
    """Return the latest numeric model version for a registered model, if any."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        logger.warning("Could not search model versions for %s: %s", model_name, exc)
        return None

    if not versions:
        return None

    return max(int(version.version) for version in versions)


def _build_model_uri_candidates(client: mlflow.MlflowClient, run_id: str, model_name: str) -> list[str]:
    """Create prioritized model URIs from most stable to least stable."""
    candidates: list[str] = []

    production_model_name = "churn-prediction-test1-prod"
    candidates.append(f"models:/{production_model_name}@production")
    candidates.append(f"models:/{model_name}@challenger")

    latest_version = _get_latest_model_version(client, model_name)
    if latest_version is not None:
        candidates.append(f"models:/{model_name}/{latest_version}")

    if MLFLOW_MODEL_URI:
        candidates.append(MLFLOW_MODEL_URI)

    if run_id:
        candidates.append(f"runs:/{run_id}/model")

    candidates.append(f"models:/{model_name}/1")

    # Keep order and drop duplicates.
    return list(dict.fromkeys(candidates))


def load_mlflow_model(run_id: str, model_name: str):
    """Load MLflow model with local-artifact-first strategy for cloud-safe inference."""
    try:
        logger.info("Trying to load model from local mlruns artifacts first")
        return _load_local_artifact_model(run_id=run_id, model_name=model_name)
    except Exception as local_exc:
        logger.warning("Local artifact model load failed: %s", local_exc)

    if _is_streamlit_cloud() and _is_local_tracking_uri(MLFLOW_TRACKING_URI):
        raise RuntimeError(
            "Local MLflow artifacts were not loadable, and MLFLOW_TRACKING_URI points to localhost "
            "which is unreachable on Streamlit Cloud."
        )

    if _is_local_tracking_uri(MLFLOW_TRACKING_URI):
        logger.warning(
            "Skipping tracking-server lookup because MLFLOW_TRACKING_URI is local (%s) and local artifacts failed.",
            MLFLOW_TRACKING_URI,
        )
        raise RuntimeError(
            "Could not load local MLflow artifacts. Tracking URI is localhost and cannot be used as remote fallback."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.MlflowClient()
    uri_candidates = _build_model_uri_candidates(client, run_id, model_name)
    requested_family = _get_requested_model_family(model_name)

    for model_uri in uri_candidates:
        try:
            logger.info("Trying model URI: %s", model_uri)
            model = mlflow.pyfunc.load_model(model_uri)
            if not _is_loaded_model_family_match(model, requested_family):
                logger.warning(
                    "Skipping model URI %s because loaded flavor does not match requested family=%s",
                    model_uri,
                    requested_family,
                )
                continue
            return model
        except Exception as exc:
            logger.warning("Model load failed for %s: %s", model_uri, exc)

    raise RuntimeError("Could not load the MLflow model from local artifacts or tracking server.")
