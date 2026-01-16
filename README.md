# J-Nita v5.0 - Web-Based OCR Pipeline

A modern web-based OCR (Optical Character Recognition) application with image conditioning and text extraction capabilities.

## Features

- 🖼️ **Image Upload**: Drag-and-drop or click to upload images
- 🔧 **Image Conditioning**: Automatic preprocessing for optimal OCR results
- 🔍 **OCR Processing**: Extract text using state-of-the-art OCR models
- ⚙️ **Configurable**: Adjust conditioning strength, OCR models, and more
- 📱 **Responsive**: Works on desktop and mobile devices

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the backend server:**
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`

3. **Open the frontend:**
   - Open `index.html` in your web browser, or
   - Serve it with a simple HTTP server:
     ```bash
     python -m http.server 8000
     ```
     Then open `http://localhost:8000`

### Deployment

### Frontend (GitHub Pages)

1. Push the repository to GitHub
2. Go to Settings → Pages
3. Set source to `main` branch and `/ (root)` folder
4. Your site will be available at `https://yourusername.github.io/J-Nita/`

### Backend Deployment ⚠️ Required

**The frontend requires a backend API to function.** See [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md) for detailed instructions.

**Quick Start (Railway - Recommended):**
1. Sign up at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select your J-Nita repository
4. Railway auto-detects and deploys
5. Copy the URL and add it to "Backend Configuration" in the web app

**Other Options:**
- **Render**: Free tier available, see [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md)
- **Heroku**: Paid plans, see [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md)
- **PythonAnywhere**: Free tier available, see [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md)

#### Option 2: Full Stack Deployment

**Backend Options:**
- **Heroku**: Deploy `app.py` as a Flask app
- **Railway**: Deploy with `requirements.txt`
- **Render**: Deploy as a web service
- **PythonAnywhere**: Upload and configure as a web app

**Frontend:**
- Deploy `index.html` to any static hosting (GitHub Pages, Netlify, Vercel)
- Update `API_BASE` in `index.html` to point to your backend URL

## Project Structure

```
J-Nita/
├── index.html              # Main web interface
├── app.py                  # Flask backend API
├── image_conditioner.py    # Image processing engine
├── ocr_workflow.py         # OCR workflow logic
├── orientation_validator.py # Orientation detection
├── config.json             # Default configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## API Endpoints

### `GET /api/health`
Health check endpoint.

### `POST /api/condition`
Condition an image for OCR.

**Request:**
```json
{
  "image": "data:image/png;base64,...",
  "config": {
    "strength": 10,
    "png_compression": 0
  }
}
```

**Response:**
```json
{
  "conditioned_image": "data:image/png;base64,...",
  "success": true
}
```

### `POST /api/ocr`
Process OCR on a conditioned image.

**Request:**
```json
{
  "image": "data:image/png;base64,...",
  "model": "Nanonets-OCR2-3B",
  "fallback_models": ["Chandra-OCR"],
  "enable_fallback": true
}
```

**Response:**
```json
{
  "raw_text": "Extracted text...",
  "markdown_text": "Formatted text...",
  "model_used": "Nanonets-OCR2-3B",
  "success": true
}
```

### `GET /api/config`
Get current configuration.

### `POST /api/config`
Save configuration.

## Configuration

The application uses `config.json` for default settings. You can:
- Modify it directly
- Use the configuration panel in the web interface
- Override settings via API calls

## Supported Image Formats

- PNG
- JPEG/JPG
- BMP
- TIFF
- WEBP
- HEIC/HEIF (if pillow-heif is installed)

## OCR Models

- **Nanonets-OCR2-3B**: High quality, medium speed (default)
- **Chandra-OCR**: Medium quality, fast
- **Dots.OCR**: Medium quality, fast
- **olmOCR-2-7B-1025**: High quality, slower

## Usage

1. **Upload Image**: Click the upload area or drag and drop an image
2. **Condition Image**: Click "Condition Image" to preprocess the image
3. **Run OCR**: Click "Run OCR" to extract text
4. **Configure**: Expand the configuration section to adjust settings

## Development

### Running Locally

```bash
# Terminal 1: Backend
python app.py

# Terminal 2: Frontend (optional, if using a server)
python -m http.server 8000
```

### Environment Variables

- `FLASK_ENV`: Set to `development` for debug mode
- `PORT`: Server port (default: 5000)

## Troubleshooting

### Backend won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that port 5000 is not in use

### CORS errors
- Make sure `flask-cors` is installed
- Check that the frontend URL matches the backend CORS settings

### OCR API errors
- Verify internet connection (uses Hugging Face API)
- Check that `gradio-client` is properly installed
- Some models may be temporarily unavailable

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
