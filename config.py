from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Config:
    project_name: str = "session_1"
    input_dir: Path = Path("data")

    # Pipeline parameters
    n_questions_per_file: int = 1
    chunk_size: int = 5000
    chunk_overlap: int = 100
    n_page_summary: int = 3

    # LLM parameters
    api_key: str = ""
    base_url: str = "https://api.perplexity.ai"

    # Safety flags
    delete_existing: bool = False  # Set True to wipe previous run

    # Derived paths -----------------------------------------------------------
    @property
    def project_dir(self) -> Path:
        return Path(self.project_name)

    @property
    def chunks_dir(self) -> Path:
        return self.project_dir / "chunks"

    @property
    def metadata_file(self) -> Path:
        return self.project_dir / "metadata.json"
    
    @property
    def question_file(self) -> Path:
        return self.project_dir / "questions.csv"