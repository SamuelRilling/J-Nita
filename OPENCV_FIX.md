# OpenCV Railway Deployment Fix

## Problem

Railway deployment was failing with:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

This error occurs because `opencv-python` requires GUI libraries (libGL) which are not available in server environments.

## Solution

Changed `opencv-python` to `opencv-python-headless` in `requirements.txt`.

### Why opencv-python-headless?

- ✅ No GUI dependencies (libGL, libX11, etc.)
- ✅ Perfect for server/container environments
- ✅ Same functionality for image processing
- ✅ Smaller installation size
- ✅ Designed specifically for headless servers

### What Changed

**Before:**
```
opencv-python>=4.8.0
```

**After:**
```
opencv-python-headless>=4.8.0
```

## Verification

After deploying with the updated `requirements.txt`:
1. Railway build should succeed ✅
2. Application should start without errors ✅
3. All OpenCV functions work identically ✅

## Note

The `opencv-python-headless` package provides the same core functionality as `opencv-python`, just without GUI-related dependencies. All image processing operations (which is what J-Nita uses) work exactly the same.
