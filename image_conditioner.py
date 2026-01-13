"""
Image Conditioning Pipeline for OCR
Handles PDF/image input, page detection, cropping, resizing, and adaptive enhancement
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image, ExifTags
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageConditioner:
    """
    Conditions input images for optimal OCR performance.
    
    Handles:
    - PDF to image conversion
    - Page detection and cropping
    - Intelligent resizing
    - Adaptive thresholding for text enhancement
    - Morphological operations for noise reduction

    Tunable parameters are exposed via the constructor and CLI to allow
    balancing aggressive cleanup vs preserving original appearance. A
    global `strength` parameter (0-100) blends between the original
    (resized) image and the fully processed image: 0 = original, 100 = fully
    processed (current behavior).
    """
    
    # Default OCR API optimal parameters
    DEFAULT_TARGET_WIDTH = 1280  # pixels
    DEFAULT_TARGET_HEIGHT = 1792  # pixels
    DEFAULT_MAX_DIMENSION = 1792
    
    # Default adaptive thresholding parameters
    DEFAULT_ADAPTIVE_BLOCK_SIZE = 11  # Kernel size for local threshold calculation (must be odd)
    DEFAULT_ADAPTIVE_C = 2  # Constant subtracted from mean
    
    # Default morphological operations
    DEFAULT_MORPH_KERNEL_SIZE = (2, 2)
    DEFAULT_MORPH_ITERATIONS = 1
    
    # Default page detection parameters
    DEFAULT_CANNY_THRESHOLD1 = 50
    DEFAULT_CANNY_THRESHOLD2 = 150
    DEFAULT_MIN_CONTOUR_AREA = 10000  # pixels^2, filters noise
    
    # Image reading extensions
    SUPPORTED_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.heic', '.heif'}
    
    def __init__(
        self,
        input_folder: str = "Input",
        output_folder: str = "results_conditioned",
        *,
        strength: int = 100,
        adaptive_block_size: Optional[int] = None,
        adaptive_C: Optional[int] = None,
        morph_kernel_size: Optional[Tuple[int, int]] = None,
        morph_iterations: Optional[int] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        canny_threshold1: Optional[int] = None,
        canny_threshold2: Optional[int] = None,
        min_contour_area: Optional[int] = None,
        max_pages: Optional[int] = None,
        denoise_ksize: Optional[int] = None,
        png_compression: Optional[int] = None,
        min_scale_for_zero: Optional[float] = None,
        debug_mode: Optional[bool] = None,
    ):
        """
        max_pages: Optional[int] - If provided and PDF has more pages, limits processing to that many pages
        """
        """
        Initialize the conditioner.
        
        Args:
            input_folder: Source folder containing images/PDFs
            output_folder: Destination folder for conditioned images
            strength: Global strength tuner (0=original, 100=full processing)
            adaptive_block_size: Adaptive threshold block size (odd integer)
            adaptive_C: Adaptive threshold C parameter (int)
            morph_kernel_size: Morph kernel size tuple (w,h)
            morph_iterations: Number of morphological iterations
            target_width: Target width in pixels
            target_height: Target height in pixels
            canny_threshold1/2: Canny thresholds for edge detection
            min_contour_area: Minimum area to consider as page contour
        """
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Global strength (0-100)
        self.strength = max(0, min(100, int(strength)))

        # Thresholding parameters
        self.ADAPTIVE_BLOCK_SIZE = adaptive_block_size or self.DEFAULT_ADAPTIVE_BLOCK_SIZE
        # ensure odd
        if self.ADAPTIVE_BLOCK_SIZE % 2 == 0:
            self.ADAPTIVE_BLOCK_SIZE += 1
        self.ADAPTIVE_C = adaptive_C if adaptive_C is not None else self.DEFAULT_ADAPTIVE_C

        # Morphological parameters
        self.MORPH_KERNEL_SIZE = morph_kernel_size or self.DEFAULT_MORPH_KERNEL_SIZE
        self.MORPH_ITERATIONS = morph_iterations if morph_iterations is not None else self.DEFAULT_MORPH_ITERATIONS

        # Resizing targets
        self.TARGET_WIDTH = target_width or self.DEFAULT_TARGET_WIDTH
        self.TARGET_HEIGHT = target_height or self.DEFAULT_TARGET_HEIGHT

        # Page detection
        self.CANNY_THRESHOLD1 = canny_threshold1 if canny_threshold1 is not None else self.DEFAULT_CANNY_THRESHOLD1
        self.CANNY_THRESHOLD2 = canny_threshold2 if canny_threshold2 is not None else self.DEFAULT_CANNY_THRESHOLD2
        self.MIN_CONTOUR_AREA = min_contour_area if min_contour_area is not None else self.DEFAULT_MIN_CONTOUR_AREA
        
        # Optional max pages to process from PDFs
        self.max_pages = max_pages if max_pages is not None else None
        
        # Optional denoising kernel size (median blur). Must be odd >=3 or None
        self.denoise_ksize = denoise_ksize if denoise_ksize and denoise_ksize >= 3 else None
        if isinstance(self.denoise_ksize, int) and self.denoise_ksize % 2 == 0:
            self.denoise_ksize += 1

        # PNG compression level (0=none, 9=max). Default: 2 (low compression to preserve detail)
        self.png_compression = 2 if png_compression is None else max(0, min(9, int(png_compression)))

        # When strength == 0, do not downscale below this fraction (0.1 - 1.0). Default: 0.5
        self.min_scale_for_zero = 0.5 if min_scale_for_zero is None else max(0.1, min(1.0, float(min_scale_for_zero)))
        logger.info(f"PNG compression: {self.png_compression}, min_scale_for_zero: {self.min_scale_for_zero}")

        # Debug mode
        self.debug_mode = debug_mode if debug_mode is not None else False
        
        # Create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        logger.info(f"Output folder: {self.output_folder}")
        
        # Create debug folders if debug mode is enabled
        if self.debug_mode:
            self.debug_bb_folder = os.path.join(self.output_folder, "my_images_bb")
            self.debug_warped_folder = os.path.join(self.output_folder, "my_images_warped")
            os.makedirs(self.debug_bb_folder, exist_ok=True)
            os.makedirs(self.debug_warped_folder, exist_ok=True)
            logger.info(f"Debug mode enabled: BB folder={self.debug_bb_folder}, Warped folder={self.debug_warped_folder}")
        else:
            self.debug_bb_folder = None
            self.debug_warped_folder = None
        
        # Store current filename for debug purposes
        self.current_filename = None
        
        logger.info(f"TUNERS: strength={self.strength}, block={self.ADAPTIVE_BLOCK_SIZE}, C={self.ADAPTIVE_C}, morph_kernel={self.MORPH_KERNEL_SIZE}, morph_iter={self.MORPH_ITERATIONS}")
    
    def process_all_files(self) -> dict:
        """
        Process all files in input folder.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        if not os.path.exists(self.input_folder):
            logger.error(f"Input folder not found: {self.input_folder}")
            return stats
        
        files = os.listdir(self.input_folder)
        logger.info(f"Found {len(files)} files in {self.input_folder}")
        
        for filename in sorted(files):
            stats['total'] += 1
            file_path = os.path.join(self.input_folder, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            try:
                if filename.lower().endswith('.pdf'):
                    self._process_pdf(file_path, filename)
                    stats['processed'] += 1
                elif any(filename.lower().endswith(ext) for ext in self.SUPPORTED_FORMATS):
                    self._process_image(file_path, filename)
                    stats['processed'] += 1
                else:
                    logger.warning(f"Skipped unsupported format: {filename}")
                    stats['skipped'] += 1
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                stats['failed'] += 1
                stats['errors'].append((filename, str(e)))
        
        return stats
    
    def _process_pdf(self, pdf_path: str, filename: str) -> None:
        """
        Convert PDF to images and process each page.
        
        Args:
            pdf_path: Full path to PDF file
            filename: Original filename (for naming output)
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image not installed. Install with: pip install pdf2image"
            )
        
        logger.info(f"Processing PDF: {filename}")
        
        pages = None
        # Try using pdf2image with Poppler (pdftoppm) if available
        try:
            poppler_path = os.environ.get('POPPLER_PATH')  # User can set this to Poppler bin dir
            # If pdftoppm is available on PATH, pdf2image will use it automatically
            from shutil import which
            pdftoppm_found = which('pdftoppm') is not None
            if poppler_path:
                pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path, first_page=1, last_page=self.max_pages)
            elif pdftoppm_found:
                pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=self.max_pages)
            else:
                # Simulate failure to trigger fallback
                raise RuntimeError("Poppler (pdftoppm) not found in PATH and POPPLER_PATH not set")

            logger.info(f"  Converted {len(pages)} pages from PDF (pdf2image + Poppler)")

        except Exception as e:
            # Attempt fallback using PyMuPDF (fitz) to avoid requiring Poppler on Windows
            logger.warning(f"pdf2image/poppler not available or failed: {e}")
            logger.info("Attempting fallback conversion using PyMuPDF (pymupdf) if installed")
            try:
                import fitz  # PyMuPDF
                import io
                from PIL import Image

                doc = fitz.open(pdf_path)
                pages = []
                for p in doc:
                    pix = p.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pages.append(pil_img)

                logger.info(f"  Converted {len(pages)} pages from PDF (PyMuPDF)")

            except ImportError:
                logger.error(
                    f"PDF conversion failed for {filename}: {e}\n"
                    "Install Poppler and ensure 'pdftoppm' is in your PATH, set POPPLER_PATH environment variable, "
                    "or install PyMuPDF with: pip install pymupdf"
                )
                raise
            except Exception as e2:
                logger.error(f"PDF conversion failed for {filename}: {e2}")
                raise
        
        # Process pages if conversion succeeded
        if not pages:
            logger.error(f"PDF conversion failed for {filename}: No pages extracted")
            raise RuntimeError("No pages extracted from PDF")

        base_name = os.path.splitext(filename)[0]
        
        for page_num, page_image in enumerate(pages, 1):
            # Convert PIL image to numpy array (BGR for OpenCV)
            cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
            
            # Store filename for debug mode (include page number for PDFs)
            self.current_filename = f"{base_name}_page{page_num}"
            
            # Process the page
            conditioned = self._condition_image(cv_image, self.current_filename)
            
            # Save with page number
            output_filename = f"{base_name}_page{page_num}_conditioned.png"
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Save PNG with configured compression (may be halved for strength==0)
            save_params = self._get_png_save_params()
            cv2.imwrite(output_path, conditioned, save_params)
            logger.info(f"  Saved page {page_num}: {output_filename} (png_compression={save_params[1]})")
    
    def _process_image(self, image_path: str, filename: str) -> None:
        """
        Process a single image file.
        
        Args:
            image_path: Full path to image file
            filename: Original filename
        """
        logger.info(f"Processing image: {filename}")
        
        # Store base filename for debug mode (without extension)
        base_name = os.path.splitext(filename)[0]
        self.current_filename = base_name
        
        # Read image, with HEIC support and PIL fallback
        image = self._read_image(image_path)
        # If the image is rotated, make it upright before processing
        upright = self._ensure_upright(image, image_path)
        if upright is not image:
            logger.info("  Image was rotated upright before processing")
        logger.info(f"  Input size: {upright.shape[1]}x{upright.shape[0]} px")
        
        # Process the image
        conditioned = self._condition_image(upright, base_name)
        
        # Save output
        output_filename = f"{filename.split('.')[0]}_conditioned.png"
        output_path = os.path.join(self.output_folder, output_filename)
        
        # Save PNG with configured compression (may be halved for strength==0)
        save_params = self._get_png_save_params()
        cv2.imwrite(output_path, conditioned, save_params)
        logger.info(f"  Output size: {conditioned.shape[1]}x{conditioned.shape[0]} px")
        logger.info(f"  Saved: {output_filename} (png_compression={save_params[1]})")
    
    def _condition_image(self, image: np.ndarray, filename: Optional[str] = None) -> np.ndarray:
        """
        Apply full conditioning pipeline to an image.
        
        Pipeline:
        1. Detect and crop to page boundaries
        2. Resize maintaining aspect ratio
        3. Adaptive thresholding
        4. Morphological operations
        
        Args:
            image: OpenCV BGR image (numpy array)
            filename: Optional filename for debug mode (uses self.current_filename if not provided)
            
        Returns:
            Conditioned image (binary/enhanced)
        """
        # Use provided filename or fall back to stored filename
        debug_filename = filename if filename is not None else self.current_filename
        
        # Step 1: Detect page boundaries and crop
        cropped = self._detect_and_crop_page(image, debug_filename)
        if cropped is None:
            logger.warning("  Page detection failed, using full image")
            cropped = image
        
        # Step 2: Resize to optimal dimensions
        resized = self._resize_image(cropped)
        
        # If strength == 0: return the minimally-processed (resized) image and skip any filters
        if self.strength <= 0:
            logger.info("    Strength=0: returning minimally-processed resized image (no denoise/threshold/morph)")
            # Return color resized image to preserve original detail
            return resized

        # Step 3: Adaptive thresholding
        # Convert to grayscale first
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Optional denoising before thresholding (skip when strength == 0)
        if self.denoise_ksize and self.strength > 0:
            gray = cv2.medianBlur(gray, self.denoise_ksize)
            logger.debug(f"    Applied median blur (ksize={self.denoise_ksize})")

        # Apply adaptive thresholding
        c_eff = self.ADAPTIVE_C
        if self.strength <= 10:
            # At low strengths, bias toward making faint dark pixels turn white to reduce speckle
            c_eff = max(1, int(self.ADAPTIVE_C * 4))
            logger.debug(f"    Low strength: increasing adaptive C to {c_eff} to reduce dark speckles")

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=self.ADAPTIVE_BLOCK_SIZE,
            C=c_eff
        )
        
        # Step 4: Morphological operations (noise reduction)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL_SIZE)
        
        # Small erode to remove noise
        binary = cv2.erode(binary, kernel, iterations=self.MORPH_ITERATIONS)
        
        # Small dilate to restore stroke thickness
        binary = cv2.dilate(binary, kernel, iterations=self.MORPH_ITERATIONS)

        # Closing to fill small dark speckles inside white areas
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=max(1, self.MORPH_ITERATIONS))
        
        # Step 5: Blend with original (strength 0-100)
        # If strength == 100: keep fully processed (binary)
        # If 0 < strength < 100: blend between gray and binary
        if self.strength >= 100:
            final = binary
        else:
            alpha = self.strength / 100.0
            # Blend binary (0/255) with gray (0-255)
            blended = (alpha * binary.astype('float32') + (1.0 - alpha) * gray.astype('float32'))
            final = np.clip(blended, 0, 255).astype('uint8')
        
        return final
    
    def _find_yellow_white_box(self, image: np.ndarray, filename: Optional[str] = None) -> Optional[Tuple[int,int,int,int]]:
        """Locate a large yellow/white rectangular board commonly used as a page background in photos.

        Returns a padded bbox as (x1,y1,x2,y2) or None if nothing confident is found.
        The method is conservative (pads and expands to include nearby edges) to avoid cutting content.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # White mask: low S, high V
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        # Yellow mask (typical paper/card backing ranges)
        lower_yellow = np.array([15, 80, 120])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Clean up the mask to form a solid region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        # Require at least a modest fraction of the image
        if area < image_area * 0.05:
            return None

        x, y, w, h = cv2.boundingRect(best)
        # Conservative pad to avoid cutting content
        pad = int(max(0.03 * max(w, h), 30))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        # Expand to include any nearby edges that might indicate clipped content
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.CANNY_THRESHOLD1, self.CANNY_THRESHOLD2)
        margin = int(0.05 * min(image.shape[0], image.shape[1]))

        def _expand_if_edges(x1, y1, x2, y2):
            # If there are edge pixels just outside any side, expand the box to include them
            nonlocal edges
            # top
            y_scan_start = max(0, y1 - margin)
            y_scan_end = y1
            if y_scan_end - y_scan_start > 0 and edges[y_scan_start:y_scan_end, x1:x2].sum() > 0:
                y1 = y_scan_start
            # bottom
            y_scan_start = y2
            y_scan_end = min(image.shape[0], y2 + margin)
            if y_scan_end - y_scan_start > 0 and edges[y_scan_start:y_scan_end, x1:x2].sum() > 0:
                y2 = y_scan_end
            # left
            x_scan_start = max(0, x1 - margin)
            x_scan_end = x1
            if x_scan_end - x_scan_start > 0 and edges[y1:y2, x_scan_start:x_scan_end].sum() > 0:
                x1 = x_scan_start
            # right
            x_scan_start = x2
            x_scan_end = min(image.shape[1], x2 + margin)
            if x_scan_end - x_scan_start > 0 and edges[y1:y2, x_scan_start:x_scan_end].sum() > 0:
                x2 = x_scan_end
            return x1, y1, x2, y2

        x1, y1, x2, y2 = _expand_if_edges(x1, y1, x2, y2)

        # Save bounding box debug image if debug mode is enabled
        if self.debug_mode and filename and self.debug_bb_folder:
            try:
                debug_img = image.copy()
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                debug_path = os.path.join(self.debug_bb_folder, f"{filename}_bb.png")
                cv2.imwrite(debug_path, debug_img)
                logger.debug(f"    Saved yellow/white box debug image: {debug_path}")
            except Exception as e:
                logger.debug(f"    Failed to save yellow/white box debug image: {e}")

        return (x1, y1, x2, y2)

    def _detect_and_crop_page(self, image: np.ndarray, filename: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Detect page boundaries using trapezoid detection with perspective correction,
        yellow/white box detection, or edge-based contours.
        Eliminates background/margins and removes bleed-through artifacts.
        
        Args:
            image: Input BGR image
            filename: Optional filename for debug mode
            
        Returns:
            Cropped/warped image or None if detection fails
        """
        # First, try trapezoid detection with perspective correction (best for skewed pages)
        quad = self._find_page_trapezoid(image)
        if quad is not None:
            try:
                warped = self._warp_to_rectangle(image, quad, filename)
                if warped is not None and warped.size > 0:
                    logger.info("  Detected trapezoid and applied perspective correction")
                    return warped
            except Exception as e:
                logger.debug(f"    Trapezoid warping failed: {e}, falling back to other methods")
        
        # Second, try yellow/white rectangular board detection (simple rectangular crop)
        box = self._find_yellow_white_box(image, filename)
        if box is not None:
            x1, y1, x2, y2 = box
            cropped = image[y1:y2, x1:x2]
            if cropped is not None and cropped.size > 0:
                logger.info("  Cropped using yellow/white bounding box")
                return cropped

        # Fallback: edge-based contour detection (previous behavior)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.CANNY_THRESHOLD1, self.CANNY_THRESHOLD2)
        
        # Dilate edges to connect gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area and find the largest rectangular region
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.MIN_CONTOUR_AREA]
        
        if not valid_contours:
            return None
        
        # Get the largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Approximate to rectangle
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Validate the box (should be reasonably sized)
        area = w * h
        image_area = image.shape[0] * image.shape[1]
        aspect = w / float(h) if h else 0.0
        # Reject tiny boxes or boxes with extreme aspect ratios
        if area < image_area * 0.25:  # Too small — be conservative to avoid bad crops
            logger.debug(f"    Detected contour too small for cropping (area {area} < {image_area*0.25:.0f})")
            return None
        if aspect < 0.2 or aspect > 5.0:
            logger.debug(f"    Detected contour aspect ratio extreme: {aspect:.2f}, skipping crop")
            return None
        
        # Crop with small margin
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        # Save bounding box debug image if debug mode is enabled
        if self.debug_mode and filename and self.debug_bb_folder:
            try:
                debug_img = image.copy()
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Also draw the contour
                cv2.drawContours(debug_img, [largest_contour], -1, (255, 0, 0), 2)
                debug_path = os.path.join(self.debug_bb_folder, f"{filename}_bb.png")
                cv2.imwrite(debug_path, debug_img)
                logger.debug(f"    Saved bounding box debug image: {debug_path}")
            except Exception as e:
                logger.debug(f"    Failed to save bounding box debug image: {e}")
        
        cropped = image[y1:y2, x1:x2]
        
        return cropped if cropped.size > 0 else None
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to optimal dimensions while preserving aspect ratio.
        
        Target: Fit within 1280x1792, maintaining aspect ratio.
        This balances OCR quality (readable text) with API efficiency.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit within target dimensions
        base_scale = min(self.TARGET_WIDTH / w, self.TARGET_HEIGHT / h, 1.0)
        scale = base_scale
        
        # If strength == 0, avoid aggressive downscaling: respect min_scale_for_zero
        if self.strength <= 0:
            scale = max(base_scale, self.min_scale_for_zero)
            logger.debug(f"    Strength=0 active: adjusted scale {base_scale:.3f}→{scale:.3f} (min_scale_for_zero)")
        
        # If original is smaller, don't upscale (preserve quality)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Use high-quality downsampling
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            logger.debug(f"    Resized: {w}x{h} → {new_w}x{new_h} (scale: {scale:.2f})")
            return resized
        else:
            logger.debug(f"    Original size maintained: {w}x{h} (≤ target)")
            return image

    def _get_png_save_params(self) -> list:
        """Return cv2.imwrite png compression parameters, halving when strength==0."""
        comp = int(self.png_compression)
        if self.strength <= 0:
            halved = max(0, int(round(comp / 2)))
            logger.info(f"  Strength=0 active: halving PNG compression {comp}→{halved}")
            comp = halved
        return [cv2.IMWRITE_PNG_COMPRESSION, comp]

    def _read_image(self, image_path: str) -> np.ndarray:
        """Read image from disk, with HEIC/HEIF support and PIL fallback."""
        ext = Path(image_path).suffix.lower()
        if ext in ('.heic', '.heif'):
            # Try pillow-heif first (registers as PIL opener)
            try:
                import pillow_heif  # type: ignore
                pillow_heif.register_heif_opener()
                pil = Image.open(image_path).convert('RGB')
            except Exception:
                # Fall back to pyheif if pillow-heif not available
                try:
                    import pyheif  # type: ignore
                    heif = pyheif.read(image_path)
                    pil = Image.frombytes('RGB', (heif.width, heif.height), heif.data, 'raw', heif.mode)
                except Exception as e:
                    raise RuntimeError(
                        "Cannot decode HEIC image. Install 'pillow-heif' or 'pyheif' to add HEIC support"
                    ) from e
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Regular path: try cv2 first, then PIL fallback
        img = cv2.imread(image_path)
        if img is None:
            try:
                pil = Image.open(image_path).convert('RGB')
                return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"Could not read image: {image_path}")
        return img

    def _ensure_upright(self, image: np.ndarray, file_path: Optional[str] = None) -> np.ndarray:
        """Try EXIF orientation, then auto-detect rotation using edge projection variance.

        Returns the rotated image (or original if no better orientation found).
        """
        # Try EXIF orientation if file_path provided
        try:
            if file_path:
                pil = Image.open(file_path)
                exif = pil._getexif()
                if exif:
                    orient_key = None
                    for k, v in ExifTags.TAGS.items():
                        if v == 'Orientation':
                            orient_key = k
                            break
                    if orient_key and orient_key in exif:
                        o = exif[orient_key]
                        if o == 3:
                            return cv2.rotate(image, cv2.ROTATE_180)
                        elif o == 6:
                            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif o == 8:
                            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        except Exception:
            pass

        # Auto-detect rotation by looking for horizontal text lines (projection variance)
        return self._auto_detect_rotation(image)

    def _auto_detect_rotation(self, image: np.ndarray) -> np.ndarray:
        """Auto-detect and correct rotation by finding orientation with strongest horizontal structure.
        
        Tests all 4 orientations, validates with text readability check, and chooses the best readable orientation.
        Uses lightweight text orientation validator to ensure text is actually readable.
        """
        try:
            from orientation_validator import OrientationValidator
            validator = OrientationValidator()
            use_validator = True
        except ImportError:
            logger.warning("    OrientationValidator not available, using basic rotation detection only")
            use_validator = False
        
        scores = {}
        validation_results = {}
        
        # Test all 4 orientations
        for angle in (0, 90, 180, 270):
            if angle == 0:
                rot = image
            elif angle == 90:
                rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rot = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rot = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Calculate variance-based score (original method)
            gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            row_sum = edges.sum(axis=1).astype('float32')
            col_sum = edges.sum(axis=0).astype('float32')
            row_var = float(np.var(row_sum))
            col_var = float(np.var(col_sum))
            variance_score = row_var - col_var
            
            # Validate text orientation (if validator available)
            readability_score = 0.0
            if use_validator:
                try:
                    is_readable, confidence, reason = validator.validate_orientation(rot)
                    readability_score = confidence if is_readable else (1.0 - confidence)
                    validation_results[angle] = (is_readable, confidence, reason)
                except Exception as e:
                    logger.debug(f"    Validation failed for {angle}°: {e}")
                    readability_score = 0.5  # Neutral score if validation fails
                    validation_results[angle] = (True, 0.5, "Validation error")
            else:
                readability_score = 0.5  # Neutral if no validator
                validation_results[angle] = (True, 0.5, "Validator not available")
            
            scores[angle] = (variance_score, readability_score, rot)
        
        # Normalize variance scores to 0-1 range for fair comparison
        var_scores = [s[0] for s in scores.values()]
        min_var = min(var_scores)
        max_var = max(var_scores)
        var_range = max_var - min_var if max_var != min_var else 1.0
        
        # Calculate combined scores (weighted: 40% variance, 60% readability)
        # Prioritize readability over variance structure
        combined_scores = {}
        for angle, (var_score, read_score, rot) in scores.items():
            normalized_var = (var_score - min_var) / var_range if var_range > 0 else 0.5
            combined = normalized_var * 0.4 + read_score * 0.6  # Readability is more important
            combined_scores[angle] = (combined, var_score, read_score, rot)
        
        # Get original (0°) scores
        orig_combined, orig_var, orig_read, _ = combined_scores[0]
        
        best_angle = 0
        best_combined = orig_combined
        
        # Find best orientation based on combined score
        for angle, (combined, var_score, read_score, rot) in combined_scores.items():
            if combined > best_combined:
                best_combined = combined
                best_angle = angle
        
        # Only rotate if:
        # 1. Best orientation is different from original
        # 2. Significant improvement (>= 0.15) OR best orientation is readable and original is not
        if best_angle != 0:
            best_comb, best_var, best_read, best_rot = combined_scores[best_angle]
            is_readable, confidence, reason = validation_results[best_angle]
            orig_readable, orig_conf, orig_reason = validation_results[0]
            
            improvement = best_combined - orig_combined
            
            # Check if best orientation is readable
            if use_validator and not is_readable:
                logger.warning(f"    Best variance score at {best_angle}° but validation indicates unreadable text")
                # Try opposite orientation (180° offset)
                opposite_angle = (best_angle + 180) % 360
                if opposite_angle in validation_results:
                    opp_readable, opp_conf, opp_reason = validation_results[opposite_angle]
                    if opp_readable and opp_conf > confidence:
                        logger.info(f"    Using opposite orientation {opposite_angle}° (validated as readable, conf={opp_conf:.2f})")
                        return combined_scores[opposite_angle][3]
            
            # STRICT ROTATION LOGIC: Only rotate if:
            # 1. Best orientation is actually readable (if validator available), AND
            # 2. Significant improvement (>= 0.15) OR best is readable and original is not
            should_rotate = False
            rotate_reason = ""
            
            if use_validator:
                # With validator: REQUIRE readability
                if not is_readable:
                    should_rotate = False
                    rotate_reason = f"Best orientation {best_angle}° not readable (conf={confidence:.2f}, {reason})"
                    logger.warning(f"    Auto-rotation REJECTED: {best_angle}° - {rotate_reason}")
                elif not orig_readable:
                    # Original not readable, best is readable - rotate
                    should_rotate = True
                    rotate_reason = f"Original not readable, {best_angle}° is readable (conf={confidence:.2f})"
                elif improvement >= 0.15 and best_read > orig_read + 0.10:
                    # Both readable, but best is significantly better
                    should_rotate = True
                    rotate_reason = f"Readability improved: {best_read:.2f} vs {orig_read:.2f} (improvement: {improvement:.2f})"
                else:
                    should_rotate = False
                    rotate_reason = f"Best {best_angle}° readable but insufficient improvement (read: {best_read:.2f} vs {orig_read:.2f}, comb: {improvement:.2f})"
                    logger.debug(f"    Auto-rotation skipped: {best_angle}° - {rotate_reason}")
            else:
                # Without validator: use combined score improvement only (fallback)
                if improvement >= 0.15:
                    should_rotate = True
                    rotate_reason = f"Combined improvement: {improvement:.2f} (no validator available)"
                else:
                    should_rotate = False
                    rotate_reason = f"Insufficient improvement: {improvement:.2f} < 0.15"
                    logger.debug(f"    Auto-rotation skipped: {best_angle}° - {rotate_reason}")
            
            if should_rotate:
                logger.info(f"    Auto-rotation: {best_angle}° (var={best_var:.0f}, readable={is_readable}, conf={confidence:.2f}, {reason}, {rotate_reason})")
                return best_rot
        
        # Return original if no significant improvement found
        if use_validator:
            orig_readable, orig_conf, orig_reason = validation_results[0]
            logger.debug(f"    Keeping original orientation (readable={orig_readable}, conf={orig_conf:.2f})")
        
        return image

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as (tl, tr, br, bl) given a (4,2) array."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # tl
        rect[2] = pts[np.argmax(s)]  # br
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # tr
        rect[3] = pts[np.argmax(diff)]  # bl
        return rect

    def _find_page_trapezoid(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Attempt to find a four-corner trapezoid/page region using aggressive edge detection.

        Returns ordered 4x2 numpy array of corner points (tl, tr, br, bl) in image coords, or None.
        """
        h, w = image.shape[:2]
        image_area = h * w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # First, try a color-mask based quadrilateral from yellow/white detection (more robust for board backgrounds)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        lower_yellow = np.array([15, 80, 120])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        km = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, km, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, km, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.02:
                continue
            peri = cv2.arcLength(cnt, True)
            found = False
            for eps in (0.005, 0.01, 0.02, 0.04):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    quads.append((area, pts))
                    found = True
                    break
            if not found:
                # try convex hull -> minAreaRect -> box points as fallback quad
                hull = cv2.convexHull(cnt)
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)
                if box_area > image_area * 0.02:
                    quads.append((area, box))
        if quads:
            quads.sort(key=lambda x: x[0], reverse=True)
            ordered = self._order_points(quads[0][1].astype('float32'))
            logger.info("    Trapezoid detected via color-mask contour (with minArea fallback)")
            return ordered

        # Aggressive Canny parameters to catch faint corners
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.01:  # skip tiny contours
                continue
            peri = cv2.arcLength(cnt, True)
            # try several epsilon values to approximate quadrilaterals
            for eps in (0.01, 0.02, 0.04, 0.08):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    quads.append((area, pts))
                    break

        if quads:
            # Choose the largest quadrilateral
            quads.sort(key=lambda x: x[0], reverse=True)
            best_pts = quads[0][1].astype('float32')
            ordered = self._order_points(best_pts)
            logger.info("    Trapezoid detected via contours")
            return ordered

        # Fallback: Hough-based line intersection method
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=min(w, h) // 8, maxLineGap=30)
        if lines is None:
            return None

        # Collect line angles and cluster into four dominant directions
        lines = [l[0] for l in lines]
        angles = [np.arctan2(float(y2) - float(y1), float(x2) - float(x1)) for (x1, y1, x2, y2) in lines]
        # Convert to degrees and normalize to [0,180)
        degs = [abs(np.degrees(a)) % 180 for a in angles]
        # KMeans-like simple clustering into two angle groups (vertical/horizontal)
        groups = {0: [], 1: []}
        for i, d in enumerate(degs):
            groups[int(d > 45)].append(lines[i])

        if not groups[0] or not groups[1]:
            return None

        # Estimate intersections between lines from different groups and pick four extreme points
        inters = []
        def intersect(l1, l2):
            x1,y1,x2,y2 = [float(v) for v in l1]
            x3,y3,x4,y4 = [float(v) for v in l2]
            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if abs(denom) < 1e-6:
                return None
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            if px < -w*0.1 or px > w*1.1 or py < -h*0.1 or py > h*1.1:
                return None
            return (int(round(px)), int(round(py)))

        for l1 in groups[0][:8]:
            for l2 in groups[1][:8]:
                p = intersect(l1, l2)
                if p is not None:
                    inters.append(p)

        if not inters:
            return None

        # Compute convex hull of intersections and approximate to 4 points
        pts = np.array(inters, dtype='int32')
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4:
            ordered = self._order_points(approx.reshape(4, 2).astype('float32'))
            logger.info("    Trapezoid detected via Hough intersections")
            return ordered

        return None

    def _warp_to_rectangle(self, image: np.ndarray, quad: np.ndarray, filename: Optional[str] = None) -> np.ndarray:
        """Warp the quadrilateral region to a straight rectangle."""
        rect = self._order_points(quad)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        # Ensure at least a small size
        maxWidth = max(100, maxWidth)
        maxHeight = max(100, maxHeight)
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype='float32')
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        # Save warped debug image if debug mode is enabled
        if self.debug_mode and filename and self.debug_warped_folder:
            try:
                debug_path = os.path.join(self.debug_warped_folder, f"{filename}_warped.png")
                cv2.imwrite(debug_path, warped)
                logger.debug(f"    Saved warped debug image: {debug_path}")
            except Exception as e:
                logger.debug(f"    Failed to save warped debug image: {e}")
        
        return warped


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Image conditioning pipeline for OCR')
    parser.add_argument('--input', default='Input', help='Input folder with images/PDFs')
    parser.add_argument('--output', default='results_conditioned', help='Output folder for conditioned images')
    parser.add_argument('--strength', type=int, default=100, help='Global strength 0-100 (0=original, 100=full processing)')
    parser.add_argument('--block-size', type=int, default=None, help='Adaptive threshold block size (odd)')
    parser.add_argument('--C', type=int, default=None, help='Adaptive threshold C parameter')
    parser.add_argument('--morph-kernel', type=int, nargs=2, metavar=('W','H'), default=None, help='Morph kernel size (W H)')
    parser.add_argument('--morph-iter', type=int, default=None, help='Morphological iterations')
    parser.add_argument('--width', type=int, default=None, help='Target width')
    parser.add_argument('--height', type=int, default=None, help='Target height')
    parser.add_argument('--canny1', type=int, default=None, help='Canny threshold1')
    parser.add_argument('--canny2', type=int, default=None, help='Canny threshold2')
    parser.add_argument('--min-area', type=int, default=None, help='Min contour area')
    parser.add_argument('--max-pages', type=int, default=None, help='Limit number of pages to process from PDFs')
    parser.add_argument('--denoise', type=int, default=None, help='Median denoise kernel size (odd, >=3)')
    parser.add_argument('--png-compression', type=int, default=2, help='PNG compression 0-9 (0=none, 9=max)')
    parser.add_argument('--min-scale-for-zero', type=float, default=None, help='Min scale fraction for strength==0 (0.1-1.0)')
    return parser.parse_args()


def main():
    """Main entry point for testing."""
    args = parse_args()

    morph_kernel = tuple(args.morph_kernel) if args.morph_kernel else None

    conditioner = ImageConditioner(
        input_folder=args.input,
        output_folder=args.output,
        strength=args.strength,
        adaptive_block_size=args.block_size,
        adaptive_C=args.C,
        morph_kernel_size=morph_kernel,
        morph_iterations=args.morph_iter,
        target_width=args.width,
        target_height=args.height,
        canny_threshold1=args.canny1,
        canny_threshold2=args.canny2,
        min_contour_area=args.min_area,
        max_pages=args.max_pages,
        denoise_ksize=args.denoise,
        png_compression=args.png_compression,
        min_scale_for_zero=args.min_scale_for_zero
    )
    
    logger.info("=" * 60)
    logger.info("IMAGE CONDITIONING PIPELINE")
    logger.info("=" * 60)
    
    stats = conditioner.process_all_files()
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Total: {stats['total']} | Processed: {stats['processed']} | "
                f"Failed: {stats['failed']} | Skipped: {stats['skipped']}")
    
    if stats['errors']:
        logger.warning("Errors encountered:")
        for filename, error in stats['errors']:
            logger.warning(f"  - {filename}: {error}")
    
    logger.info(f"Output saved to: {conditioner.output_folder}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
