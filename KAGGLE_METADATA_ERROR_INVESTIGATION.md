# KAGGLE METADATA_ERROR INVESTIGATION

## Error Summary

**Error:** `TypeError: open() got an unexpected keyword argument 'metadata_errors'`

**Status:** NOT FOUND IN CODEBASE — This error does not originate from Clips Kitty source code.

## Investigation Results

### Codebase Search
- **Grep for `metadata_errors`**: No matches in any `.py` file
- **Grep for `open(.*errors=`**: All uses valid values (`"replace"`, `"ignore"`)
- **All `open()` calls inspected**: 89 files checked — none pass `metadata_errors`

### Valid Python `open()` `errors` Parameters
```python
# Valid values only:
'strict', 'ignore', 'replace', 'xmlcharrefreplace',
'backslashreplace', 'namereplace', 'surrogateescape', 'surrogatepass'
```
`metadata_errors` is **not a valid parameter**.

## Likely Runtime Sources on Kaggle

| Library | Version Risk | Why |
|---------|-------------|-----|
| `faster-whisper` ≥ 1.2.0 | Medium | Uses CTranslate2 which loads model files; may pass invalid args in some versions |
| `ultralytics` ≥ 8.4.0 | Medium | Downloads/loads YOLO weights; internal file handling |
| `onnxruntime` ≥ 1.26 | Low | Loads ONNX models; C++ backend may surface Python errors oddly |
| `yt-dlp` ≥ 2026.8 | Low | Writes metadata JSON; uses standard `open()` |
| `huggingface-hub` | Low | Model downloads; uses `open()` for cache |

## Debugging Steps for Kaggle

### 1. Add Traceback Capture
In the notebook (Cell 17), wrap the pipeline call:

```python
import traceback
import sys

try:
    result = adapter.process_url(VIDEO_URL, force=False)
except Exception as e:
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR MESSAGE: {e}")
    traceback.print_exc()
    
    # Check if it's the metadata_errors error
    if "metadata_errors" in str(e):
        print("\n>>> THIS IS THE metadata_errors BUG <<<")
        # Print call stack to identify source
        for frame in traceback.extract_tb(e.__traceback__):
            print(f"  File {frame.filename}, line {frame.lineno}, in {frame.name}")
            print(f"    {frame.line}")
```

### 2. Check Library Versions
Add to Cell 5 (after imports):
```python
import faster_whisper
import ultralytics
import onnxruntime
import yt_dlp
import huggingface_hub

print(f"faster-whisper: {faster_whisper.__version__}")
print(f"ultralytics: {ultralytics.__version__}")
print(f"onnxruntime: {onnxruntime.__version__}")
print(f"yt-dlp: {yt_dlp.version.__version__}")
print(f"huggingface-hub: {huggingface_hub.__version__}")
```

### 3. Isolate the Stage
The error likely occurs during one of these stages:
- **Download** (yt-dlp writing metadata)
- **Transcription** (faster-whisper loading model)
- **Analysis** (ultralytics loading YOLO/TalkNet)
- **Render** (ffmpeg/encoding)

Add stage tracking to identify which stage fails.

## Potential Fixes

### Fix 1: Pin Compatible Versions
Update `requirements-kaggle.txt` with known-working versions:
```txt
faster-whisper==1.1.1   # Avoid 1.2.x if it has this bug
ultralytics==8.3.0      # Avoid 8.4+ if it has this bug
onnxruntime==1.25.0     # Avoid 1.26+ if it has this bug
```

### Fix 2: Monkey-Patch open() (Last Resort)
Add to Cell 1 (before imports):
```python
# Debug: trace all open() calls with metadata_errors
import builtins
_original_open = builtins.open

def traced_open(*args, **kwargs):
    if 'metadata_errors' in kwargs:
        import traceback
        print(">>> open() called with metadata_errors <<<")
        traceback.print_stack()
        kwargs.pop('metadata_errors')  # Remove invalid arg
    return _original_open(*args, **kwargs)

builtins.open = traced_open
```

### Fix 3: Upgrade/Downgrade Specific Library
If traceback shows `faster-whisper`:
```bash
!pip install faster-whisper==1.1.1 --force-reinstall
```
If traceback shows `ultralytics`:
```bash
!pip install ultralytics==8.3.0 --force-reinstall
```

## Root Cause Hypothesis

This is likely a **version incompatibility** between:
- Python 3.11/3.12 on Kaggle
- A library version that passes `metadata_errors` to `open()` (possibly a typo in the library itself)
- The library's internal file handling for model downloads/cache

The error `metadata_errors` suggests a typo for `errors=` with a metadata-related value, possibly from:
- `errors="metadata_replace"` (typo)
- `errors="metadata_ignore"` (typo)
- A library parameter named `metadata_errors` incorrectly passed to `open()`

## Next Steps

1. **Run the notebook with traceback capture** (Step 1 above)
2. **Identify the exact library and line** from the traceback
3. **Pin or upgrade that library** in `requirements-kaggle.txt`
4. **Report upstream** if it's a library bug

## Verification Checklist

- [ ] Run notebook with traceback capture
- [ ] Identify failing stage (download/transcribe/analyze/render)
- [ ] Identify failing library from traceback
- [ ] Test with pinned library versions
- [ ] Update requirements-kaggle.txt with working versions
- [ ] Verify end-to-end pipeline completes

---

**Note:** This error has **not been observed in the local Windows environment** — it appears to be specific to the Kaggle container environment and library versions available there.