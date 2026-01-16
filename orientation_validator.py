<<<<<<< HEAD
"""
Lightweight text orientation validator
Checks if text in an image is readable without heavy OCR
Uses character pattern analysis and simple heuristics
"""

import cv2
import numpy as np
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class OrientationValidator:
    """
    Validates text orientation using lightweight methods:
    1. Character stroke pattern analysis (horizontal vs vertical)
    2. Text line density analysis
    3. Simple word pattern detection (if available)
    """
    
    def __init__(self):
        # Common readable English words for validation (frequent short words)
        self.common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
            'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'man'
        }
    
    def validate_orientation(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Validate if image text is correctly oriented (readable).
        
        Args:
            image: BGR or grayscale image to validate
            
        Returns:
            Tuple of (is_readable, confidence_score, reason)
            - is_readable: True if text appears correctly oriented
            - confidence_score: 0.0-1.0 confidence in the assessment
            - reason: Human-readable explanation
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize size for consistent analysis (resize if too large)
        h, w = gray.shape
        if h * w > 2000000:  # If larger than ~1.4MP, resize for speed
            scale = np.sqrt(2000000 / (h * w))
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        
        scores = []
        reasons = []
        
        # Method 1: Horizontal text line detection
        line_score, line_reason = self._detect_horizontal_lines(gray)
        scores.append(line_score)
        reasons.append(line_reason)
        
        # Method 2: Character stroke pattern (horizontal strokes should dominate)
        stroke_score, stroke_reason = self._analyze_stroke_patterns(gray)
        scores.append(stroke_score)
        reasons.append(stroke_reason)
        
        # Method 3: Text density gradient (should be stronger top-to-bottom)
        density_score, density_reason = self._analyze_text_density_gradient(gray)
        scores.append(density_score)
        reasons.append(density_reason)
        
        # Method 4: Baseline and character orientation detection (NEW - critical for upside-down detection)
        baseline_score, baseline_reason = self._detect_baseline_orientation(gray)
        scores.append(baseline_score)
        reasons.append(baseline_reason)
        
        # Weighted combination
        # Baseline detection is most critical for upside-down (weight: 0.45)
        # Line detection is reliable (weight: 0.25)
        # Stroke patterns are secondary (weight: 0.20)
        # Density gradient is least important (weight: 0.10)
        final_score = (
            scores[0] * 0.25 +  # line_score
            scores[1] * 0.20 +  # stroke_score
            scores[2] * 0.10 +  # density_score
            scores[3] * 0.45    # baseline_score (NEW - most important)
        )
        
        is_readable = final_score >= 0.70  # Raised threshold to 0.70 for very strict validation (prevents false positives on upside-down text)
        confidence = min(abs(final_score - 0.70) * 5, 1.0)  # Confidence based on distance from stricter threshold
        
        # Combine reasons
        primary_reason = reasons[np.argmax(scores)]
        summary = f"Score: {final_score:.2f} (line:{scores[0]:.2f}, stroke:{scores[1]:.2f}, density:{scores[2]:.2f}, baseline:{scores[3]:.2f})"
        
        return is_readable, confidence, summary
    
    def _detect_horizontal_lines(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Detect horizontal text lines - correctly oriented text should have clear horizontal lines.
        
        Returns:
            Score (0-1) and reason string
        """
        # Apply adaptive threshold to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Horizontal projection (sum along rows)
        h_projection = binary.sum(axis=1)
        
        # Find peaks in horizontal projection (text lines)
        # Smooth the projection first
        kernel_size = max(5, int(gray.shape[0] / 50))
        if kernel_size % 2 == 0:
            kernel_size += 1
        h_smooth = cv2.GaussianBlur(h_projection.reshape(-1, 1), (kernel_size, 1), 0).flatten()
        
        # Count significant peaks (text lines)
        mean_val = h_smooth.mean()
        std_val = h_smooth.std()
        threshold = mean_val + 0.5 * std_val
        
        peaks = []
        for i in range(1, len(h_smooth) - 1):
            if h_smooth[i] > threshold and h_smooth[i] > h_smooth[i-1] and h_smooth[i] > h_smooth[i+1]:
                peaks.append(i)
        
        # Score based on number of clear text lines
        # For readable text, we should have multiple clear horizontal lines
        h, w = gray.shape
        expected_lines = max(3, int(h / 100))  # Expect at least 1 line per 100px height, minimum 3
        
        if len(peaks) >= expected_lines * 0.6:  # At least 60% of expected lines
            score = min(1.0, len(peaks) / expected_lines)
            return score, f"Detected {len(peaks)} horizontal text lines (expected ~{expected_lines})"
        else:
            score = len(peaks) / (expected_lines * 0.6) if expected_lines > 0 else 0.0
            return max(0.0, min(1.0, score)), f"Only {len(peaks)} horizontal lines detected (expected ~{expected_lines})"
    
    def _analyze_stroke_patterns(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Analyze character stroke patterns - horizontal strokes should dominate in correctly oriented text.
        
        Returns:
            Score (0-1) and reason string
        """
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal vs vertical edges using gradient
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitudes
        grad_x = np.abs(sobelx)
        grad_y = np.abs(sobely)
        
        # In correctly oriented text:
        # - Horizontal strokes (like in 'e', 'a', 't') should create vertical edges (strong grad_x)
        # - Vertical strokes (like in 'l', 'i', 't') should create horizontal edges (strong grad_y)
        # But overall, we should have more horizontal text lines, so more vertical edges
        
        # Count significant horizontal vs vertical gradients
        threshold_h = grad_x.mean() + grad_x.std()
        threshold_v = grad_y.mean() + grad_y.std()
        
        strong_h_edges = np.sum(grad_x > threshold_h)
        strong_v_edges = np.sum(grad_y > threshold_v)
        
        total_strong = strong_h_edges + strong_v_edges
        if total_strong == 0:
            return 0.5, "Insufficient edge information"
        
        # In correctly oriented text, vertical edges (from horizontal strokes) should dominate
        # But not too much - some balance is expected
        h_ratio = strong_h_edges / total_strong
        
        # Optimal range: 0.45-0.65 (more vertical edges but not extreme)
        if 0.45 <= h_ratio <= 0.65:
            score = 1.0 - abs(h_ratio - 0.55) * 2  # Best at 0.55
        elif h_ratio < 0.45:
            # Too many horizontal edges (might be rotated 90° or upside down)
            score = h_ratio / 0.45
        else:
            # Too many vertical edges (might be sideways)
            score = (1.0 - h_ratio) / 0.35
        
        return max(0.0, min(1.0, score)), f"Horizontal edge ratio: {h_ratio:.2f} (optimal: 0.45-0.65)"
    
    def _analyze_text_density_gradient(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Analyze text density gradient - correctly oriented text should have consistent top-to-bottom flow.
        
        Returns:
            Score (0-1) and reason string
        """
        # Apply threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Divide image into horizontal bands
        h, w = gray.shape
        num_bands = 8
        band_height = h // num_bands
        
        densities = []
        for i in range(num_bands):
            y_start = i * band_height
            y_end = (i + 1) * band_height if i < num_bands - 1 else h
            band = binary[y_start:y_end, :]
            density = band.sum() / (band.size * 255.0)  # Normalized density
            densities.append(density)
        
        # Calculate gradient (change in density from top to bottom)
        # In correctly oriented text, density should vary somewhat but not chaotically
        gradients = np.diff(densities)
        
        # Good text should have moderate variation (not all same, not chaotic)
        gradient_var = np.var(gradients)
        gradient_mean = np.abs(np.mean(gradients))
        
        # Optimal: moderate variation (some change between lines), small mean (no strong trend)
        # Too little variation: might be upside down or blank
        # Too much variation: might be sideways or corrupted
        if gradient_var < 0.001:  # Too uniform
            score = 0.3
            reason = f"Density too uniform (var={gradient_var:.4f}) - possibly upside down"
        elif gradient_var > 0.01:  # Too chaotic
            score = 0.4
            reason = f"Density too chaotic (var={gradient_var:.4f}) - possibly sideways"
        else:
            score = 1.0 - (gradient_var - 0.001) / 0.009  # Best around 0.001-0.005
            reason = f"Good density variation (var={gradient_var:.4f})"
        
        return max(0.0, min(1.0, score)), reason
    
    def _detect_baseline_orientation(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Detect baseline orientation - correctly oriented text has baseline at bottom of characters.
        This is critical for detecting upside-down text, which has baseline at top.
        
        In correctly oriented text:
        - Baseline (where most characters sit) should be in lower-middle portion
        - Ascenders (b, d, h, k, l, t) extend above baseline
        - Descenders (g, j, p, q, y) extend below baseline
        - Text density should be bottom-heavy (more pixels in lower half)
        
        Returns:
            Score (0-1) and reason string
        """
        # Apply threshold to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        h, w = gray.shape
        
        # Method 1: Analyze vertical density distribution
        # For correctly oriented text, baseline should be in lower-middle region
        vertical_projection = binary.sum(axis=1)  # Sum of pixels per row
        
        # Find the row with maximum density (likely baseline)
        max_row_idx = np.argmax(vertical_projection)
        baseline_position = max_row_idx / h  # Normalized position (0=top, 1=bottom)
        
        # In correctly oriented text, baseline should be in lower-middle (0.45-0.75 of height)
        # If baseline is in upper portion (<0.45), text is likely upside-down
        if 0.45 <= baseline_position <= 0.75:
            position_score = 1.0
            position_reason = f"Baseline at {baseline_position:.2f} (optimal range)"
        elif baseline_position < 0.45:
            # Baseline too high - likely upside-down (STRICT REJECTION)
            # Scale more harshly: position < 0.3 gets very low score
            if baseline_position < 0.3:
                position_score = 0.1  # Very likely upside-down
            else:
                position_score = 0.3 + (baseline_position - 0.3) / 0.15 * 0.2  # 0.3-0.5 range
            position_reason = f"Baseline too high at {baseline_position:.2f} (likely upside-down)"
        else:  # baseline_position > 0.75
            # Baseline too low - might still be readable but unusual
            position_score = max(0.5, 1.0 - ((baseline_position - 0.75) / 0.25))  # Penalize if >0.75
            position_reason = f"Baseline very low at {baseline_position:.2f}"
        
        # Method 2: Bottom-heaviness check
        # Correctly oriented text should have more pixels in lower half
        upper_half = binary[:h//2, :].sum()
        lower_half = binary[h//2:, :].sum()
        total_pixels = upper_half + lower_half
        
        if total_pixels > 0:
            lower_ratio = lower_half / total_pixels
            # Good text: 55-70% of pixels in lower half
            if 0.55 <= lower_ratio <= 0.70:
                heaviness_score = 1.0
                heaviness_reason = f"Bottom-heavy: {lower_ratio:.2f}"
            elif lower_ratio < 0.55:
                # Top-heavy - likely upside-down
                heaviness_score = lower_ratio / 0.55
                heaviness_reason = f"Top-heavy: {lower_ratio:.2f} (likely upside-down)"
            else:  # lower_ratio > 0.70
                heaviness_score = 1.0 - ((lower_ratio - 0.70) / 0.30)
                heaviness_reason = f"Extremely bottom-heavy: {lower_ratio:.2f}"
        else:
            heaviness_score = 0.5
            heaviness_reason = "Insufficient text pixels"
        
        # Method 3: Check for ascender/descender pattern
        # In correctly oriented text, we should see more vertical extension above baseline (ascenders)
        # than in upside-down text
        if baseline_position < 1.0:
            baseline_row = int(max_row_idx)
            # Analyze region above baseline (ascenders) vs below baseline (descenders)
            ascender_region = binary[:baseline_row, :] if baseline_row > 0 else np.zeros((1, w), dtype=np.uint8)
            descender_region = binary[baseline_row:, :] if baseline_row < h else np.zeros((1, w), dtype=np.uint8)
            
            ascender_density = ascender_region.sum() / (ascender_region.size * 255.0) if ascender_region.size > 0 else 0
            descender_density = descender_region.sum() / (descender_region.size * 255.0) if descender_region.size > 0 else 0
            
            # In correctly oriented text, ascenders should have reasonable density (letters extend up)
            # Upside-down text will have unusual pattern - ascenders will be very weak
            # Also check ratio: ascenders should be stronger than descenders (more letters have ascenders)
            if ascender_density > 0.05 and descender_density > 0.02:
                # Both present - check ratio
                ascender_ratio = ascender_density / (ascender_density + descender_density) if (ascender_density + descender_density) > 0 else 0
                # In correct text, ascenders should dominate (ratio 0.6-0.8)
                if 0.6 <= ascender_ratio <= 0.8:
                    pattern_score = 1.0
                    pattern_reason = f"Good ascender/descender ratio: {ascender_ratio:.2f}"
                elif ascender_ratio < 0.4:
                    # Descenders dominate - likely upside-down
                    pattern_score = 0.2
                    pattern_reason = f"Descenders dominate ({ascender_ratio:.2f}) - likely upside-down"
                else:
                    pattern_score = 0.7
                    pattern_reason = f"Unusual ratio: asc={ascender_ratio:.2f}"
            elif ascender_density < 0.02:
                # Very weak ascenders - likely upside-down
                pattern_score = 0.15
                pattern_reason = f"Very weak ascenders: {ascender_density:.3f} (likely upside-down)"
            else:
                pattern_score = 0.5
                pattern_reason = f"Unusual pattern: asc={ascender_density:.3f}, desc={descender_density:.3f}"
        else:
            pattern_score = 0.5
            pattern_reason = "Baseline detection unclear"
        
        # Combined baseline score (weighted average)
        # CRITICAL: If baseline is too high (<0.45), heavily penalize (likely upside-down)
        if baseline_position < 0.45:
            # Baseline in upper portion - strongly reject (likely upside-down)
            # Use minimum of position, heaviness, and pattern scores (all must agree)
            baseline_final = min(position_score, heaviness_score, pattern_score) * 0.7  # Additional penalty
        else:
            # Baseline in acceptable range - use weighted average
            baseline_final = (
                position_score * 0.40 +   # Position is most important
                heaviness_score * 0.35 +  # Bottom-heaviness is critical
                pattern_score * 0.25      # Pattern analysis is secondary
            )
        
        combined_reason = f"Baseline: {baseline_position:.2f}, {heaviness_reason}, {pattern_reason}"
        
        return max(0.0, min(1.0, baseline_final)), combined_reason


def detect_gibberish(text: str, min_valid_words: int = 3) -> Tuple[bool, float, str]:
    """
    Detect if OCR result text is gibberish vs readable.
    
    Uses heuristics:
    - Character frequency analysis
    - Common word detection
    - Letter-to-symbol ratio
    - Word length distribution
    
    Args:
        text: Text to analyze
        min_valid_words: Minimum number of valid words to consider text readable
        
    Returns:
        Tuple of (is_gibberish, confidence, reason)
        - is_gibberish: True if text appears to be gibberish
        - confidence: 0.0-1.0 confidence in assessment
        - reason: Explanation
    """
    if not text or len(text.strip()) < 10:
        return True, 1.0, "Text too short or empty"
    
    text_clean = text.strip()
    
    # Common readable English words (frequent words)
    common_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
        'boy', 'did', 'let', 'put', 'say', 'she', 'too', 'use', 'man', 'any',
        'ask', 'bad', 'big', 'cut', 'end', 'far', 'few', 'fly', 'got', 'hot'
    }
    
    scores = []
    reasons = []
    
    # Common word pairs (bigrams) in English
    common_bigrams = {
        'of the', 'in the', 'to the', 'for the', 'on the', 'at the', 'with the',
        'it is', 'that is', 'this is', 'there is', 'here is', 'what is',
        'you are', 'we are', 'they are', 'i am', 'he is', 'she is',
        'and the', 'but the', 'or the', 'as the', 'if the',
        'has been', 'have been', 'had been', 'will be', 'would be',
        'to be', 'for a', 'in a', 'on a', 'at a', 'with a', 'to a',
        'as a', 'an the', 'to do', 'to go', 'to see', 'to get', 'to have',
        'the and', 'the for', 'the was', 'the are', 'the not', 'the you',
        'is the', 'is a', 'is an', 'is not', 'is that', 'is this',
        'and is', 'and was', 'and the', 'and to', 'and for'
    }
    
    # Method 1: Common word detection (revised - check ratio more carefully)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text_clean.lower())
    if len(words) >= min_valid_words:
        common_count = sum(1 for w in words if w in common_words)
        word_ratio = common_count / len(words) if words else 0
        
        # Low common word ratio suggests gibberish
        # BUT: Even with valid words, if they don't form sentences, it's still gibberish
        if word_ratio < 0.05:
            word_score = 0.9  # Very low ratio = gibberish
        elif word_ratio < 0.10:
            word_score = 0.7  # Low ratio = likely gibberish
        elif word_ratio < 0.20:
            word_score = 0.5  # Medium ratio = check other factors
        else:
            word_score = 0.3  # Good ratio = likely valid (but still check coherence)
        
        scores.append(word_score)
        reasons.append(f"Found {common_count}/{len(words)} common words (ratio: {word_ratio:.2f})")
    else:
        scores.append(1.0)
        reasons.append(f"Insufficient words detected ({len(words)} < {min_valid_words})")
    
    # Method 2: Semantic coherence check (NEW - checks for meaningful sentences)
    coherence_score, coherence_reason = _check_semantic_coherence(text_clean, words, common_words)
    scores.append(coherence_score)
    reasons.append(coherence_reason)
    
    # Method 3: Bigram/trigram analysis (NEW - checks for common word pairs)
    bigram_score, bigram_reason = _check_bigrams(text_clean, common_bigrams)
    scores.append(bigram_score)
    reasons.append(bigram_reason)
    
    # Method 4: Character frequency (English should have reasonable distribution)
    text_lower = text_clean.lower()
    letters = re.findall(r'[a-z]', text_lower)
    if len(letters) >= 20:
        char_counts = {}
        for char in letters:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Common English letters should dominate (e, t, a, o, i, n)
        common_chars = {'e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r'}
        common_count = sum(char_counts.get(c, 0) for c in common_chars)
        common_ratio = common_count / len(letters)
        
        # Good English text should have 40-60% common letters
        if 0.4 <= common_ratio <= 0.6:
            char_score = 0.2  # Low gibberish score
        else:
            char_score = min(1.0, abs(common_ratio - 0.5) * 2)
        scores.append(char_score)
        reasons.append(f"Common letter ratio: {common_ratio:.2f} (optimal: 0.4-0.6)")
    else:
        scores.append(0.8)
        reasons.append(f"Too few letters for frequency analysis ({len(letters)})")
    
    # Method 5: Letter-to-symbol ratio
    total_chars = len(re.findall(r'[a-zA-Z]', text_clean))
    symbol_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text_clean))
    total_non_space = len(re.findall(r'\S', text_clean))
    
    if total_non_space > 0:
        letter_ratio = total_chars / total_non_space
        symbol_ratio = symbol_chars / total_non_space
        
        # Good text should have high letter ratio (>0.7) and low symbol ratio (<0.1)
        if letter_ratio > 0.7 and symbol_ratio < 0.1:
            ratio_score = 0.2
        elif letter_ratio < 0.5 or symbol_ratio > 0.2:
            ratio_score = 0.9  # High gibberish score
        else:
            ratio_score = 0.5
        scores.append(ratio_score)
        reasons.append(f"Letter ratio: {letter_ratio:.2f}, Symbol ratio: {symbol_ratio:.2f}")
    else:
        scores.append(1.0)
        reasons.append("No readable characters found")
    
    # Method 6: Word length distribution (should have reasonable variety)
    if words:
        word_lengths = [len(w) for w in words]
        avg_length = np.mean(word_lengths)
        length_std = np.std(word_lengths)
        
        # Average English word length is ~4.5-5.5 characters
        # Too uniform or extreme suggests issues
        if 3.5 <= avg_length <= 6.5 and length_std > 1.0:
            length_score = 0.2
        elif avg_length < 2.5 or avg_length > 8.0:
            length_score = 0.8
        else:
            length_score = 0.5
        scores.append(length_score)
        reasons.append(f"Avg word length: {avg_length:.1f}±{length_std:.1f}")
    else:
        scores.append(0.9)
        reasons.append("No words found")
    
    # Weighted average (semantic coherence and bigrams are most important for detecting incoherent text)
    final_score = (
        scores[0] * 0.20 +  # common words (reduced weight)
        scores[1] * 0.35 +  # semantic coherence (NEW - most important)
        scores[2] * 0.25 +  # bigram analysis (NEW - very important)
        scores[3] * 0.10 +  # character frequency
        scores[4] * 0.05 +  # letter-to-symbol ratio
        scores[5] * 0.05    # word length
    )
    
    is_gibberish = final_score >= 0.55  # Lowered threshold from 0.65 to 0.55 for more sensitive detection
    confidence = min(abs(final_score - 0.55) * 3, 1.0)  # Confidence based on distance from threshold
    
    primary_reason = reasons[np.argmax(scores)]
    summary = f"Gibberish score: {final_score:.2f} (threshold: 0.55) - {primary_reason}"
    
    return is_gibberish, confidence, summary


def _check_semantic_coherence(text: str, words: list, common_words: set) -> Tuple[float, str]:
    """
    Check semantic coherence - valid text should form meaningful sentences, not random words.
    
    Checks:
    - Sentence structure (capitalization, punctuation)
    - Word repetition patterns (gibberish often has unusual repetition)
    - Common word proximity (valid words should appear near other valid words)
    - Sentence length variety
    
    Returns:
        Score (0-1, lower = more gibberish) and reason
    """
    if len(words) < 5:
        return 0.5, "Insufficient words for coherence check"
    
    text_clean = text.strip()
    
    # Check 1: Sentence structure - valid text should have some capitalization and punctuation
    sentences = re.split(r'[.!?]\s+', text_clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 1:
        sentences = [text_clean]  # Treat whole text as one sentence
    
    # Check 1: Sentence structure - valid text should have proper sentence structure
    valid_structure_count = 0
    for sent in sentences:
        sent_clean = re.sub(r'^<[^>]+>', '', sent).strip()
        if sent_clean:
            first_char = sent_clean[0]
            sent_words = re.findall(r'\b[a-zA-Z]{2,}\b', sent.lower())
            # Valid sentence: has reasonable length AND starts properly (capitalization or digit)
            if (first_char.isupper() or first_char.isdigit()) and 3 <= len(sent_words) <= 30:
                valid_structure_count += 1
    
    structure_score = valid_structure_count / max(len(sentences), 1) if sentences else 0.3
    structure_score = min(1.0, structure_score)  # Cap at 1.0 to prevent overflow
    
    # Check 2: Word repetition - gibberish often has unusual patterns
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Check for excessive repetition of uncommon words (sign of gibberish)
    total_words = len(words)
    unique_words = len(word_counts)
    uniqueness_ratio = unique_words / total_words if total_words > 0 else 0
    
    # Good text: 0.6-0.9 uniqueness (some repetition is normal, too much repetition is suspicious)
    if 0.6 <= uniqueness_ratio <= 0.9:
        repetition_score = 1.0
    elif uniqueness_ratio < 0.5:  # Too much repetition
        repetition_score = 0.3
    elif uniqueness_ratio > 0.95:  # Almost all unique (might be garbled)
        repetition_score = 0.5
    else:
        repetition_score = 0.7
    
    # Check 3: Common word clustering - valid words should appear near other valid words
    # In gibberish, valid words are scattered randomly
    if len(words) >= 10:
        common_indices = [i for i, w in enumerate(words) if w in common_words]
        if len(common_indices) >= 3:
            # Check if common words cluster or are scattered
            gaps = [common_indices[i+1] - common_indices[i] for i in range(len(common_indices)-1)]
            avg_gap = np.mean(gaps) if gaps else len(words)
            max_gap = max(gaps) if gaps else len(words)
            
            # In valid text, common words should appear relatively frequently (small avg gap)
            # In gibberish, common words are rare and scattered (large gaps)
            if avg_gap <= 5 and max_gap <= 15:
                clustering_score = 1.0
            elif avg_gap <= 8 and max_gap <= 25:
                clustering_score = 0.7
            else:
                clustering_score = 0.3
        else:
            clustering_score = 0.5  # Too few common words to assess
    else:
        clustering_score = 0.5
    
    # Combined coherence score
    coherence_final = (
        structure_score * 0.35 +    # Sentence structure
        repetition_score * 0.30 +   # Word repetition patterns
        clustering_score * 0.35     # Common word clustering
    )
    
    reason = f"Coherence: struct={structure_score:.2f}, repet={repetition_score:.2f}, cluster={clustering_score:.2f}"
    
    return max(0.0, min(1.0, coherence_final)), reason


def _check_bigrams(text: str, common_bigrams: set) -> Tuple[float, str]:
    """
    Check for common word pairs (bigrams) - valid text should have common word combinations.
    
    Returns:
        Score (0-1, lower = more gibberish) and reason
    """
    text_lower = text.lower()
    
    # Extract all bigrams (word pairs)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text_lower)
    
    if len(words) < 2:
        return 0.5, "Insufficient words for bigram analysis"
    
    bigrams = []
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        bigrams.append(bigram)
    
    if not bigrams:
        return 0.5, "No bigrams found"
    
    # Count common bigrams
    common_bigram_count = sum(1 for bg in bigrams if bg in common_bigrams)
    bigram_ratio = common_bigram_count / len(bigrams) if bigrams else 0
    
    # Valid text should have some common bigrams (5-20% is reasonable)
    # Gibberish will have very few or none (even if individual words are valid)
    if bigram_ratio >= 0.05:
        score = 0.2  # Low gibberish score (has common bigrams)
    elif bigram_ratio >= 0.02:
        score = 0.5  # Medium (some common bigrams)
    else:
        score = 0.85  # High gibberish score (no common bigrams = incoherent)
    
    reason = f"Bigram ratio: {bigram_ratio:.3f} ({common_bigram_count}/{len(bigrams)} common pairs)"
    
    return score, reason
=======
"""
Lightweight text orientation validator
Checks if text in an image is readable without heavy OCR
Uses character pattern analysis and simple heuristics
"""

import cv2
import numpy as np
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class OrientationValidator:
    """
    Validates text orientation using lightweight methods:
    1. Character stroke pattern analysis (horizontal vs vertical)
    2. Text line density analysis
    3. Simple word pattern detection (if available)
    """
    
    def __init__(self):
        # Common readable English words for validation (frequent short words)
        self.common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
            'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'man'
        }
    
    def validate_orientation(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Validate if image text is correctly oriented (readable).
        
        Args:
            image: BGR or grayscale image to validate
            
        Returns:
            Tuple of (is_readable, confidence_score, reason)
            - is_readable: True if text appears correctly oriented
            - confidence_score: 0.0-1.0 confidence in the assessment
            - reason: Human-readable explanation
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize size for consistent analysis (resize if too large)
        h, w = gray.shape
        if h * w > 2000000:  # If larger than ~1.4MP, resize for speed
            scale = np.sqrt(2000000 / (h * w))
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        
        scores = []
        reasons = []
        
        # Method 1: Horizontal text line detection
        line_score, line_reason = self._detect_horizontal_lines(gray)
        scores.append(line_score)
        reasons.append(line_reason)
        
        # Method 2: Character stroke pattern (horizontal strokes should dominate)
        stroke_score, stroke_reason = self._analyze_stroke_patterns(gray)
        scores.append(stroke_score)
        reasons.append(stroke_reason)
        
        # Method 3: Text density gradient (should be stronger top-to-bottom)
        density_score, density_reason = self._analyze_text_density_gradient(gray)
        scores.append(density_score)
        reasons.append(density_reason)
        
        # Method 4: Baseline and character orientation detection (NEW - critical for upside-down detection)
        baseline_score, baseline_reason = self._detect_baseline_orientation(gray)
        scores.append(baseline_score)
        reasons.append(baseline_reason)
        
        # Weighted combination
        # Baseline detection is most critical for upside-down (weight: 0.45)
        # Line detection is reliable (weight: 0.25)
        # Stroke patterns are secondary (weight: 0.20)
        # Density gradient is least important (weight: 0.10)
        final_score = (
            scores[0] * 0.25 +  # line_score
            scores[1] * 0.20 +  # stroke_score
            scores[2] * 0.10 +  # density_score
            scores[3] * 0.45    # baseline_score (NEW - most important)
        )
        
        is_readable = final_score >= 0.70  # Raised threshold to 0.70 for very strict validation (prevents false positives on upside-down text)
        confidence = min(abs(final_score - 0.70) * 5, 1.0)  # Confidence based on distance from stricter threshold
        
        # Combine reasons
        primary_reason = reasons[np.argmax(scores)]
        summary = f"Score: {final_score:.2f} (line:{scores[0]:.2f}, stroke:{scores[1]:.2f}, density:{scores[2]:.2f}, baseline:{scores[3]:.2f})"
        
        return is_readable, confidence, summary
    
    def _detect_horizontal_lines(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Detect horizontal text lines - correctly oriented text should have clear horizontal lines.
        
        Returns:
            Score (0-1) and reason string
        """
        # Apply adaptive threshold to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Horizontal projection (sum along rows)
        h_projection = binary.sum(axis=1)
        
        # Find peaks in horizontal projection (text lines)
        # Smooth the projection first
        kernel_size = max(5, int(gray.shape[0] / 50))
        if kernel_size % 2 == 0:
            kernel_size += 1
        h_smooth = cv2.GaussianBlur(h_projection.reshape(-1, 1), (kernel_size, 1), 0).flatten()
        
        # Count significant peaks (text lines)
        mean_val = h_smooth.mean()
        std_val = h_smooth.std()
        threshold = mean_val + 0.5 * std_val
        
        peaks = []
        for i in range(1, len(h_smooth) - 1):
            if h_smooth[i] > threshold and h_smooth[i] > h_smooth[i-1] and h_smooth[i] > h_smooth[i+1]:
                peaks.append(i)
        
        # Score based on number of clear text lines
        # For readable text, we should have multiple clear horizontal lines
        h, w = gray.shape
        expected_lines = max(3, int(h / 100))  # Expect at least 1 line per 100px height, minimum 3
        
        if len(peaks) >= expected_lines * 0.6:  # At least 60% of expected lines
            score = min(1.0, len(peaks) / expected_lines)
            return score, f"Detected {len(peaks)} horizontal text lines (expected ~{expected_lines})"
        else:
            score = len(peaks) / (expected_lines * 0.6) if expected_lines > 0 else 0.0
            return max(0.0, min(1.0, score)), f"Only {len(peaks)} horizontal lines detected (expected ~{expected_lines})"
    
    def _analyze_stroke_patterns(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Analyze character stroke patterns - horizontal strokes should dominate in correctly oriented text.
        
        Returns:
            Score (0-1) and reason string
        """
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal vs vertical edges using gradient
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitudes
        grad_x = np.abs(sobelx)
        grad_y = np.abs(sobely)
        
        # In correctly oriented text:
        # - Horizontal strokes (like in 'e', 'a', 't') should create vertical edges (strong grad_x)
        # - Vertical strokes (like in 'l', 'i', 't') should create horizontal edges (strong grad_y)
        # But overall, we should have more horizontal text lines, so more vertical edges
        
        # Count significant horizontal vs vertical gradients
        threshold_h = grad_x.mean() + grad_x.std()
        threshold_v = grad_y.mean() + grad_y.std()
        
        strong_h_edges = np.sum(grad_x > threshold_h)
        strong_v_edges = np.sum(grad_y > threshold_v)
        
        total_strong = strong_h_edges + strong_v_edges
        if total_strong == 0:
            return 0.5, "Insufficient edge information"
        
        # In correctly oriented text, vertical edges (from horizontal strokes) should dominate
        # But not too much - some balance is expected
        h_ratio = strong_h_edges / total_strong
        
        # Optimal range: 0.45-0.65 (more vertical edges but not extreme)
        if 0.45 <= h_ratio <= 0.65:
            score = 1.0 - abs(h_ratio - 0.55) * 2  # Best at 0.55
        elif h_ratio < 0.45:
            # Too many horizontal edges (might be rotated 90° or upside down)
            score = h_ratio / 0.45
        else:
            # Too many vertical edges (might be sideways)
            score = (1.0 - h_ratio) / 0.35
        
        return max(0.0, min(1.0, score)), f"Horizontal edge ratio: {h_ratio:.2f} (optimal: 0.45-0.65)"
    
    def _analyze_text_density_gradient(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Analyze text density gradient - correctly oriented text should have consistent top-to-bottom flow.
        
        Returns:
            Score (0-1) and reason string
        """
        # Apply threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Divide image into horizontal bands
        h, w = gray.shape
        num_bands = 8
        band_height = h // num_bands
        
        densities = []
        for i in range(num_bands):
            y_start = i * band_height
            y_end = (i + 1) * band_height if i < num_bands - 1 else h
            band = binary[y_start:y_end, :]
            density = band.sum() / (band.size * 255.0)  # Normalized density
            densities.append(density)
        
        # Calculate gradient (change in density from top to bottom)
        # In correctly oriented text, density should vary somewhat but not chaotically
        gradients = np.diff(densities)
        
        # Good text should have moderate variation (not all same, not chaotic)
        gradient_var = np.var(gradients)
        gradient_mean = np.abs(np.mean(gradients))
        
        # Optimal: moderate variation (some change between lines), small mean (no strong trend)
        # Too little variation: might be upside down or blank
        # Too much variation: might be sideways or corrupted
        if gradient_var < 0.001:  # Too uniform
            score = 0.3
            reason = f"Density too uniform (var={gradient_var:.4f}) - possibly upside down"
        elif gradient_var > 0.01:  # Too chaotic
            score = 0.4
            reason = f"Density too chaotic (var={gradient_var:.4f}) - possibly sideways"
        else:
            score = 1.0 - (gradient_var - 0.001) / 0.009  # Best around 0.001-0.005
            reason = f"Good density variation (var={gradient_var:.4f})"
        
        return max(0.0, min(1.0, score)), reason
    
    def _detect_baseline_orientation(self, gray: np.ndarray) -> Tuple[float, str]:
        """
        Detect baseline orientation - correctly oriented text has baseline at bottom of characters.
        This is critical for detecting upside-down text, which has baseline at top.
        
        In correctly oriented text:
        - Baseline (where most characters sit) should be in lower-middle portion
        - Ascenders (b, d, h, k, l, t) extend above baseline
        - Descenders (g, j, p, q, y) extend below baseline
        - Text density should be bottom-heavy (more pixels in lower half)
        
        Returns:
            Score (0-1) and reason string
        """
        # Apply threshold to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        h, w = gray.shape
        
        # Method 1: Analyze vertical density distribution
        # For correctly oriented text, baseline should be in lower-middle region
        vertical_projection = binary.sum(axis=1)  # Sum of pixels per row
        
        # Find the row with maximum density (likely baseline)
        max_row_idx = np.argmax(vertical_projection)
        baseline_position = max_row_idx / h  # Normalized position (0=top, 1=bottom)
        
        # In correctly oriented text, baseline should be in lower-middle (0.45-0.75 of height)
        # If baseline is in upper portion (<0.45), text is likely upside-down
        if 0.45 <= baseline_position <= 0.75:
            position_score = 1.0
            position_reason = f"Baseline at {baseline_position:.2f} (optimal range)"
        elif baseline_position < 0.45:
            # Baseline too high - likely upside-down (STRICT REJECTION)
            # Scale more harshly: position < 0.3 gets very low score
            if baseline_position < 0.3:
                position_score = 0.1  # Very likely upside-down
            else:
                position_score = 0.3 + (baseline_position - 0.3) / 0.15 * 0.2  # 0.3-0.5 range
            position_reason = f"Baseline too high at {baseline_position:.2f} (likely upside-down)"
        else:  # baseline_position > 0.75
            # Baseline too low - might still be readable but unusual
            position_score = max(0.5, 1.0 - ((baseline_position - 0.75) / 0.25))  # Penalize if >0.75
            position_reason = f"Baseline very low at {baseline_position:.2f}"
        
        # Method 2: Bottom-heaviness check
        # Correctly oriented text should have more pixels in lower half
        upper_half = binary[:h//2, :].sum()
        lower_half = binary[h//2:, :].sum()
        total_pixels = upper_half + lower_half
        
        if total_pixels > 0:
            lower_ratio = lower_half / total_pixels
            # Good text: 55-70% of pixels in lower half
            if 0.55 <= lower_ratio <= 0.70:
                heaviness_score = 1.0
                heaviness_reason = f"Bottom-heavy: {lower_ratio:.2f}"
            elif lower_ratio < 0.55:
                # Top-heavy - likely upside-down
                heaviness_score = lower_ratio / 0.55
                heaviness_reason = f"Top-heavy: {lower_ratio:.2f} (likely upside-down)"
            else:  # lower_ratio > 0.70
                heaviness_score = 1.0 - ((lower_ratio - 0.70) / 0.30)
                heaviness_reason = f"Extremely bottom-heavy: {lower_ratio:.2f}"
        else:
            heaviness_score = 0.5
            heaviness_reason = "Insufficient text pixels"
        
        # Method 3: Check for ascender/descender pattern
        # In correctly oriented text, we should see more vertical extension above baseline (ascenders)
        # than in upside-down text
        if baseline_position < 1.0:
            baseline_row = int(max_row_idx)
            # Analyze region above baseline (ascenders) vs below baseline (descenders)
            ascender_region = binary[:baseline_row, :] if baseline_row > 0 else np.zeros((1, w), dtype=np.uint8)
            descender_region = binary[baseline_row:, :] if baseline_row < h else np.zeros((1, w), dtype=np.uint8)
            
            ascender_density = ascender_region.sum() / (ascender_region.size * 255.0) if ascender_region.size > 0 else 0
            descender_density = descender_region.sum() / (descender_region.size * 255.0) if descender_region.size > 0 else 0
            
            # In correctly oriented text, ascenders should have reasonable density (letters extend up)
            # Upside-down text will have unusual pattern - ascenders will be very weak
            # Also check ratio: ascenders should be stronger than descenders (more letters have ascenders)
            if ascender_density > 0.05 and descender_density > 0.02:
                # Both present - check ratio
                ascender_ratio = ascender_density / (ascender_density + descender_density) if (ascender_density + descender_density) > 0 else 0
                # In correct text, ascenders should dominate (ratio 0.6-0.8)
                if 0.6 <= ascender_ratio <= 0.8:
                    pattern_score = 1.0
                    pattern_reason = f"Good ascender/descender ratio: {ascender_ratio:.2f}"
                elif ascender_ratio < 0.4:
                    # Descenders dominate - likely upside-down
                    pattern_score = 0.2
                    pattern_reason = f"Descenders dominate ({ascender_ratio:.2f}) - likely upside-down"
                else:
                    pattern_score = 0.7
                    pattern_reason = f"Unusual ratio: asc={ascender_ratio:.2f}"
            elif ascender_density < 0.02:
                # Very weak ascenders - likely upside-down
                pattern_score = 0.15
                pattern_reason = f"Very weak ascenders: {ascender_density:.3f} (likely upside-down)"
            else:
                pattern_score = 0.5
                pattern_reason = f"Unusual pattern: asc={ascender_density:.3f}, desc={descender_density:.3f}"
        else:
            pattern_score = 0.5
            pattern_reason = "Baseline detection unclear"
        
        # Combined baseline score (weighted average)
        # CRITICAL: If baseline is too high (<0.45), heavily penalize (likely upside-down)
        if baseline_position < 0.45:
            # Baseline in upper portion - strongly reject (likely upside-down)
            # Use minimum of position, heaviness, and pattern scores (all must agree)
            baseline_final = min(position_score, heaviness_score, pattern_score) * 0.7  # Additional penalty
        else:
            # Baseline in acceptable range - use weighted average
            baseline_final = (
                position_score * 0.40 +   # Position is most important
                heaviness_score * 0.35 +  # Bottom-heaviness is critical
                pattern_score * 0.25      # Pattern analysis is secondary
            )
        
        combined_reason = f"Baseline: {baseline_position:.2f}, {heaviness_reason}, {pattern_reason}"
        
        return max(0.0, min(1.0, baseline_final)), combined_reason


def detect_gibberish(text: str, min_valid_words: int = 3) -> Tuple[bool, float, str]:
    """
    Detect if OCR result text is gibberish vs readable.
    
    Uses heuristics:
    - Character frequency analysis
    - Common word detection
    - Letter-to-symbol ratio
    - Word length distribution
    
    Args:
        text: Text to analyze
        min_valid_words: Minimum number of valid words to consider text readable
        
    Returns:
        Tuple of (is_gibberish, confidence, reason)
        - is_gibberish: True if text appears to be gibberish
        - confidence: 0.0-1.0 confidence in assessment
        - reason: Explanation
    """
    if not text or len(text.strip()) < 10:
        return True, 1.0, "Text too short or empty"
    
    text_clean = text.strip()
    
    # Common readable English words (frequent words)
    common_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
        'boy', 'did', 'let', 'put', 'say', 'she', 'too', 'use', 'man', 'any',
        'ask', 'bad', 'big', 'cut', 'end', 'far', 'few', 'fly', 'got', 'hot'
    }
    
    scores = []
    reasons = []
    
    # Common word pairs (bigrams) in English
    common_bigrams = {
        'of the', 'in the', 'to the', 'for the', 'on the', 'at the', 'with the',
        'it is', 'that is', 'this is', 'there is', 'here is', 'what is',
        'you are', 'we are', 'they are', 'i am', 'he is', 'she is',
        'and the', 'but the', 'or the', 'as the', 'if the',
        'has been', 'have been', 'had been', 'will be', 'would be',
        'to be', 'for a', 'in a', 'on a', 'at a', 'with a', 'to a',
        'as a', 'an the', 'to do', 'to go', 'to see', 'to get', 'to have',
        'the and', 'the for', 'the was', 'the are', 'the not', 'the you',
        'is the', 'is a', 'is an', 'is not', 'is that', 'is this',
        'and is', 'and was', 'and the', 'and to', 'and for'
    }
    
    # Method 1: Common word detection (revised - check ratio more carefully)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text_clean.lower())
    if len(words) >= min_valid_words:
        common_count = sum(1 for w in words if w in common_words)
        word_ratio = common_count / len(words) if words else 0
        
        # Low common word ratio suggests gibberish
        # BUT: Even with valid words, if they don't form sentences, it's still gibberish
        if word_ratio < 0.05:
            word_score = 0.9  # Very low ratio = gibberish
        elif word_ratio < 0.10:
            word_score = 0.7  # Low ratio = likely gibberish
        elif word_ratio < 0.20:
            word_score = 0.5  # Medium ratio = check other factors
        else:
            word_score = 0.3  # Good ratio = likely valid (but still check coherence)
        
        scores.append(word_score)
        reasons.append(f"Found {common_count}/{len(words)} common words (ratio: {word_ratio:.2f})")
    else:
        scores.append(1.0)
        reasons.append(f"Insufficient words detected ({len(words)} < {min_valid_words})")
    
    # Method 2: Semantic coherence check (NEW - checks for meaningful sentences)
    coherence_score, coherence_reason = _check_semantic_coherence(text_clean, words, common_words)
    scores.append(coherence_score)
    reasons.append(coherence_reason)
    
    # Method 3: Bigram/trigram analysis (NEW - checks for common word pairs)
    bigram_score, bigram_reason = _check_bigrams(text_clean, common_bigrams)
    scores.append(bigram_score)
    reasons.append(bigram_reason)
    
    # Method 4: Character frequency (English should have reasonable distribution)
    text_lower = text_clean.lower()
    letters = re.findall(r'[a-z]', text_lower)
    if len(letters) >= 20:
        char_counts = {}
        for char in letters:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Common English letters should dominate (e, t, a, o, i, n)
        common_chars = {'e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r'}
        common_count = sum(char_counts.get(c, 0) for c in common_chars)
        common_ratio = common_count / len(letters)
        
        # Good English text should have 40-60% common letters
        if 0.4 <= common_ratio <= 0.6:
            char_score = 0.2  # Low gibberish score
        else:
            char_score = min(1.0, abs(common_ratio - 0.5) * 2)
        scores.append(char_score)
        reasons.append(f"Common letter ratio: {common_ratio:.2f} (optimal: 0.4-0.6)")
    else:
        scores.append(0.8)
        reasons.append(f"Too few letters for frequency analysis ({len(letters)})")
    
    # Method 5: Letter-to-symbol ratio
    total_chars = len(re.findall(r'[a-zA-Z]', text_clean))
    symbol_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text_clean))
    total_non_space = len(re.findall(r'\S', text_clean))
    
    if total_non_space > 0:
        letter_ratio = total_chars / total_non_space
        symbol_ratio = symbol_chars / total_non_space
        
        # Good text should have high letter ratio (>0.7) and low symbol ratio (<0.1)
        if letter_ratio > 0.7 and symbol_ratio < 0.1:
            ratio_score = 0.2
        elif letter_ratio < 0.5 or symbol_ratio > 0.2:
            ratio_score = 0.9  # High gibberish score
        else:
            ratio_score = 0.5
        scores.append(ratio_score)
        reasons.append(f"Letter ratio: {letter_ratio:.2f}, Symbol ratio: {symbol_ratio:.2f}")
    else:
        scores.append(1.0)
        reasons.append("No readable characters found")
    
    # Method 6: Word length distribution (should have reasonable variety)
    if words:
        word_lengths = [len(w) for w in words]
        avg_length = np.mean(word_lengths)
        length_std = np.std(word_lengths)
        
        # Average English word length is ~4.5-5.5 characters
        # Too uniform or extreme suggests issues
        if 3.5 <= avg_length <= 6.5 and length_std > 1.0:
            length_score = 0.2
        elif avg_length < 2.5 or avg_length > 8.0:
            length_score = 0.8
        else:
            length_score = 0.5
        scores.append(length_score)
        reasons.append(f"Avg word length: {avg_length:.1f}±{length_std:.1f}")
    else:
        scores.append(0.9)
        reasons.append("No words found")
    
    # Weighted average (semantic coherence and bigrams are most important for detecting incoherent text)
    final_score = (
        scores[0] * 0.20 +  # common words (reduced weight)
        scores[1] * 0.35 +  # semantic coherence (NEW - most important)
        scores[2] * 0.25 +  # bigram analysis (NEW - very important)
        scores[3] * 0.10 +  # character frequency
        scores[4] * 0.05 +  # letter-to-symbol ratio
        scores[5] * 0.05    # word length
    )
    
    is_gibberish = final_score >= 0.55  # Lowered threshold from 0.65 to 0.55 for more sensitive detection
    confidence = min(abs(final_score - 0.55) * 3, 1.0)  # Confidence based on distance from threshold
    
    primary_reason = reasons[np.argmax(scores)]
    summary = f"Gibberish score: {final_score:.2f} (threshold: 0.55) - {primary_reason}"
    
    return is_gibberish, confidence, summary


def _check_semantic_coherence(text: str, words: list, common_words: set) -> Tuple[float, str]:
    """
    Check semantic coherence - valid text should form meaningful sentences, not random words.
    
    Checks:
    - Sentence structure (capitalization, punctuation)
    - Word repetition patterns (gibberish often has unusual repetition)
    - Common word proximity (valid words should appear near other valid words)
    - Sentence length variety
    
    Returns:
        Score (0-1, lower = more gibberish) and reason
    """
    if len(words) < 5:
        return 0.5, "Insufficient words for coherence check"
    
    text_clean = text.strip()
    
    # Check 1: Sentence structure - valid text should have some capitalization and punctuation
    sentences = re.split(r'[.!?]\s+', text_clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 1:
        sentences = [text_clean]  # Treat whole text as one sentence
    
    # Check 1: Sentence structure - valid text should have proper sentence structure
    valid_structure_count = 0
    for sent in sentences:
        sent_clean = re.sub(r'^<[^>]+>', '', sent).strip()
        if sent_clean:
            first_char = sent_clean[0]
            sent_words = re.findall(r'\b[a-zA-Z]{2,}\b', sent.lower())
            # Valid sentence: has reasonable length AND starts properly (capitalization or digit)
            if (first_char.isupper() or first_char.isdigit()) and 3 <= len(sent_words) <= 30:
                valid_structure_count += 1
    
    structure_score = valid_structure_count / max(len(sentences), 1) if sentences else 0.3
    structure_score = min(1.0, structure_score)  # Cap at 1.0 to prevent overflow
    
    # Check 2: Word repetition - gibberish often has unusual patterns
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Check for excessive repetition of uncommon words (sign of gibberish)
    total_words = len(words)
    unique_words = len(word_counts)
    uniqueness_ratio = unique_words / total_words if total_words > 0 else 0
    
    # Good text: 0.6-0.9 uniqueness (some repetition is normal, too much repetition is suspicious)
    if 0.6 <= uniqueness_ratio <= 0.9:
        repetition_score = 1.0
    elif uniqueness_ratio < 0.5:  # Too much repetition
        repetition_score = 0.3
    elif uniqueness_ratio > 0.95:  # Almost all unique (might be garbled)
        repetition_score = 0.5
    else:
        repetition_score = 0.7
    
    # Check 3: Common word clustering - valid words should appear near other valid words
    # In gibberish, valid words are scattered randomly
    if len(words) >= 10:
        common_indices = [i for i, w in enumerate(words) if w in common_words]
        if len(common_indices) >= 3:
            # Check if common words cluster or are scattered
            gaps = [common_indices[i+1] - common_indices[i] for i in range(len(common_indices)-1)]
            avg_gap = np.mean(gaps) if gaps else len(words)
            max_gap = max(gaps) if gaps else len(words)
            
            # In valid text, common words should appear relatively frequently (small avg gap)
            # In gibberish, common words are rare and scattered (large gaps)
            if avg_gap <= 5 and max_gap <= 15:
                clustering_score = 1.0
            elif avg_gap <= 8 and max_gap <= 25:
                clustering_score = 0.7
            else:
                clustering_score = 0.3
        else:
            clustering_score = 0.5  # Too few common words to assess
    else:
        clustering_score = 0.5
    
    # Combined coherence score
    coherence_final = (
        structure_score * 0.35 +    # Sentence structure
        repetition_score * 0.30 +   # Word repetition patterns
        clustering_score * 0.35     # Common word clustering
    )
    
    reason = f"Coherence: struct={structure_score:.2f}, repet={repetition_score:.2f}, cluster={clustering_score:.2f}"
    
    return max(0.0, min(1.0, coherence_final)), reason


def _check_bigrams(text: str, common_bigrams: set) -> Tuple[float, str]:
    """
    Check for common word pairs (bigrams) - valid text should have common word combinations.
    
    Returns:
        Score (0-1, lower = more gibberish) and reason
    """
    text_lower = text.lower()
    
    # Extract all bigrams (word pairs)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text_lower)
    
    if len(words) < 2:
        return 0.5, "Insufficient words for bigram analysis"
    
    bigrams = []
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        bigrams.append(bigram)
    
    if not bigrams:
        return 0.5, "No bigrams found"
    
    # Count common bigrams
    common_bigram_count = sum(1 for bg in bigrams if bg in common_bigrams)
    bigram_ratio = common_bigram_count / len(bigrams) if bigrams else 0
    
    # Valid text should have some common bigrams (5-20% is reasonable)
    # Gibberish will have very few or none (even if individual words are valid)
    if bigram_ratio >= 0.05:
        score = 0.2  # Low gibberish score (has common bigrams)
    elif bigram_ratio >= 0.02:
        score = 0.5  # Medium (some common bigrams)
    else:
        score = 0.85  # High gibberish score (no common bigrams = incoherent)
    
    reason = f"Bigram ratio: {bigram_ratio:.3f} ({common_bigram_count}/{len(bigrams)} common pairs)"
    
    return score, reason
>>>>>>> 3998df6f2eb9ed25e696e30ca04cacd75174931a
