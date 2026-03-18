import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "llm_interactions.jsonl"


def salvar_resultado_llm(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    enriched_record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "diabetes_interpretation",
        "prompt_version": "v1",
        **record
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")