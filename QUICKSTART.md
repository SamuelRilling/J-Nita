# OCR5 Quick Start Guide

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the Frontend

**Option A: Direct File**
- Simply open `index.html` in your web browser
- Uses the hosted backend at `https://j-nita.onrender.com`

**Option B: Local Server**
```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000
```

## Using the Application

1. **Upload Image**: Click the upload area or drag and drop an image
2. **Condition Image**: Click "Condition Image" to preprocess
3. **Run OCR**: Click "Run OCR" to extract text
4. **Configure**: Expand configuration section to adjust settings

## Deployment

### GitHub Pages (Frontend Only)

1. Push to GitHub
2. Go to Settings > Pages
3. Select source branch (usually `main`)
4. Frontend will be available at `https://yourusername.github.io/OCR5/`

## Troubleshooting

### OCR API errors
- Requires internet connection
- Hugging Face API must be accessible
- Some models may be temporarily unavailable
