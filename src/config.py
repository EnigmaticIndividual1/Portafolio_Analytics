from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    positions_path: Path = BASE_DIR / "data" / "positions.csv"
    reports_dir: Path = BASE_DIR / "reports"
    charts_dir: Path = reports_dir / "charts"
    tables_dir: Path = reports_dir / "tables"
    history_dir: Path = reports_dir / "history"
    snapshots_csv: Path = history_dir / "daily_snapshots.csv"
    snapshots_db: Path = history_dir / "daily_snapshots.sqlite"
    historical_pnl_path: Path = BASE_DIR / "data" / "historical_pnl.tsv"

    benchmark: str = "SPY"
    history_period: str = "1y"
    history_interval: str = "1d"

    risk_free_rate_annual: float = 0.0

SETTINGS = Settings()
