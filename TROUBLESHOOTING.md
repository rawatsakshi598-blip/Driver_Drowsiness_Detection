# TROUBLESHOOTING — common errors and exact fixes

If a test or run fails, find your error below and copy-paste the fix.

---

## 1. `AttributeError: module 'mediapipe' has no attribute 'solutions'`

**Cause**: You have a newer MediaPipe (1.x) that dropped the legacy
`mp.solutions.face_mesh` API. The new code in commit `1a71d97` already
handles this automatically by falling back to the Tasks API, but the
**cleanest** fix is to install the legacy version which has the model
bundled (no download needed).

**Fix** (run this in your activated venv):

```powershell
# Windows PowerShell (in your venv):
pip install --force-reinstall mediapipe==0.10.14 "numpy<2.2" "protobuf>=4.25.3,<5" "scipy==1.13.1"
```

```bash
# Linux/macOS (in your venv):
pip install --force-reinstall mediapipe==0.10.14 'numpy<2.2' 'protobuf>=4.25.3,<5' 'scipy==1.13.1'
```

Then verify:

```bash
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"
# Expected: 0.10.14 True
```

Then re-run:

```bash
python -m pytest tests/ -v
# Expected: 75 passed
```

> **Why these specific versions?**
> - `mediapipe==0.10.14` — last version with the legacy FaceMesh API bundled.
> - `numpy<2.2` — mediapipe 0.10.14 was built against numpy 1.x; numpy 2.2+ breaks the C ABI.
> - `protobuf>=4.25.3,<5` — mediapipe 0.10.14 uses protobuf's old `label` API which was removed in protobuf 5+.
> - `scipy==1.13.1` — last scipy version that works with numpy 1.26.x.

---

## 2. `ModuleNotFoundError: No module named 'mediapipe'`

You haven't installed the requirements. Run:

```bash
pip install -r requirements.txt
```

If you see an error about `numpy` or `protobuf` after this, follow fix #1 above.

---

## 3. `cv2.error: OpenCV(4.x) ... can't open camera`

The webcam can't be opened. Try in order:

**Linux**:
```bash
# Add yourself to the video group:
sudo usermod -aG video $USER
# Log out and back in (or reboot).

# Verify the camera exists:
ls -l /dev/video*
# Should show /dev/video0 (or similar).
```

**Windows**:
Open `config.yaml` and change:
```yaml
video:
  backend: null
```
to:
```yaml
video:
  backend: 200   # CAP_DSHOW — DirectShow, faster on Windows
```

**macOS**:
System Settings → Privacy & Security → Camera → enable Terminal / VS Code.

**Wrong camera index**: Try `--source 1` or `--source 2`:
```bash
python main.py monitor --source 1
```

---

## 4. `RuntimeError: module was compiled against NumPy C-API version 0x12`

You have numpy 2.x installed but a package was compiled against numpy 1.x.
Fix:

```bash
pip install --force-reinstall "numpy==1.26.4" "scipy==1.13.1"
```

---

## 5. `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`

You have protobuf 5+ or 6+ but mediapipe needs protobuf 4.x.
Fix:

```bash
pip install --force-reinstall "protobuf==4.25.3"
```

---

## 6. `TF_ENABLE_ONEDNN_OPTS` warnings

These are **harmless** informational messages from TensorFlow. To silence them:

**Linux/macOS** (bash):
```bash
export TF_CPP_MIN_LOG_LEVEL=3
python -m pytest tests/
```

**Windows** (PowerShell):
```powershell
$env:TF_CPP_MIN_LOG_LEVEL=3
python -m pytest tests/
```

To make it permanent, add the line to your shell profile.

---

## 7. Flask server: `OSError: [Errno 98] Address already in use`

Port 8000 is already taken. Either kill the other process or use a different port:

```bash
# Find what's using port 8000:
# Linux:
sudo lsof -i :8000
# Windows:
netstat -ano | findstr :8000

# Use a different port:
python main.py server --port 8001
```

---

## 8. Tests timing out

If `tests/system/test_fps_benchmark.py` times out, your CPU is slower
than expected. Run with a longer timeout:

```bash
python -m pytest tests/system/ --timeout=300
```

Or skip the slow system tests:

```bash
python -m pytest tests/unit tests/e2e --timeout=120
```

---

## 9. `Permission denied (publickey)` when pushing

Your git credentials aren't cached. Run:

```bash
git config --global credential.helper store
echo "https://rawatsakshi598-blip:YOUR_GITHUB_PAT@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials
git push origin main
```

On Windows, use the credential manager instead:

```powershell
git config --global credential.helper manager
# The first push will pop up a window asking for your PAT.
```

---

## 10. `push declined due to repository private mode`

If the repo is private, make sure your PAT has the `repo` scope. Go to:
https://github.com/settings/tokens → regenerate the token with `repo` checked.

---

## 11. Tests fail with `ImportError: No module named 'drowsiness'`

You're running pytest from the wrong directory. Make sure you're in the
repo root (where `drowsiness/` is a subdirectory):

```bash
cd path/to/Driver_Drowsiness_Detection
ls drowsiness/   # should show __init__.py, app.py, etc.
python -m pytest tests/ -v
```

---

## 12. Cartoon-face test videos don't trigger any alarm

If you regenerated the test videos with `scripts/make_test_video.py`
and got cartoon faces instead of real ones, MediaPipe won't detect
them. The committed `tests/assets/test_*.mp4` files are built from
real face photos — don't regenerate unless you have the real face
photos in `tests/assets/face/`.

If you need to regenerate, the script `scripts/make_test_video.py`
expects these real photos:
- `tests/assets/face/face2.jpg` (alert, EAR≈0.45)
- `tests/assets/face/sleepy2.jpg` (sleepy, EAR≈0.25)
- `tests/assets/face/yawn_3bf78d5140f2.jpg` (yawning, MAR≈0.97)

These are committed to the repo. Don't delete them.

---

## Still stuck?

1. Run with verbose output:
   ```bash
   python -m pytest tests/ -v --tb=long 2>&1 | tee test_output.log
   ```
2. Open a GitHub issue at:
   https://github.com/rawatsakshi598-blip/Driver_Drowsiness_Detection/issues
3. Paste the full `test_output.log` content.
