# Changelog

## Version 5.0 - J-Nita

### Changed
- **Project renamed**: OCR5 → J-Nita
- **Version**: Set to 5.0
- **OpenCV dependency**: Changed from `opencv-python` to `opencv-python-headless` for Railway deployment compatibility
  - Fixes `libGL.so.1` missing library error
  - No functional changes, same image processing capabilities

### Fixed
- Railway deployment errors with OpenCV (libGL missing)
- Updated all references from OCR5 to J-Nita
- Updated localStorage keys to use `jnita_backend_url`

### Documentation
- Updated README.md with new project name
- Updated deployment guides
- Created OPENCV_FIX.md explaining the dependency change
