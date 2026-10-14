"""Only data-path resolution and Pillow compatibility are adapted here."""
from pathlib import Path
from PIL import Image
from gym_minigrid.minigrid import MiniGridEnv

if not hasattr(Image, "LANCZOS"):
    Image.LANCZOS = Image.Resampling.LANCZOS

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data/environments"


class LunarRoverEnv(MiniGridEnv):
    def __init__(self, site="site01", start_time=0, max_steps=960, seed=1337, data_root=None):
        if site not in ("site01", "site06", "site07"):
            raise ValueError("site must be site01, site06, or site07")
        if not isinstance(start_time, int) or not 0 <= start_time < 1440:
            raise ValueError("start_time must be an integer in [0, 1439]")
        self.data_root = Path(data_root).resolve() if data_root else DEFAULT_DATA_ROOT
        super().__init__(grid_size=202, site_name=site, start_time=start_time,
                         max_steps=max_steps, seed=seed)

    def read_environment(self, map_site, start_time):
        folder = self.data_root / map_site
        required = ("slope.csv", "Vsun.csv", "lunarSolarLight.csv", "lunarSurfaceTemp.csv")
        missing = [str(folder / name) for name in required if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing environment inputs: " + ", ".join(missing)
                                    + ". See data/README.md and scripts/download_data.py.")
        return super().read_environment(str(folder), start_time)
