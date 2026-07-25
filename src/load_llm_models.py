import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from groq import Groq


def load_groq_client_and_models() -> Tuple[Groq, str, str]:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    api_key = os.getenv("GROQ_API_KEY")
    model_predict = os.getenv("GROQ_MODEL_PREDICT")
    model_judge = os.getenv("GROQ_MODEL_JUDGE")

    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in environment")
    if not model_predict:
        raise EnvironmentError("Missing GROQ_MODEL_PREDICT in environment")
    if not model_judge:
        raise EnvironmentError("Missing GROQ_MODEL_JUDGE in environment")

    return Groq(api_key=api_key), model_predict, model_judge
