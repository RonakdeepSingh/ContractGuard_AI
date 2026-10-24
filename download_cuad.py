import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
OUT_DIR = Path("data")

print("Downloading CUAD (about 18 MB)... please wait")
raw = urllib.request.urlopen(URL, timeout=120).read()

OUT_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as zf:
    zf.extractall(OUT_DIR)
    for name in zf.namelist():
        print("Saved:", name)

print("Done! Files are in the 'data' folder.")