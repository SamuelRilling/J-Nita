# Test Plan: PDF/DOCX Export & Split-Pane UI

## 1. Overview
This test plan covers the verification of the PDF/DOCX export functionality and the Split-Pane UI redesign for the J-Nita OCR tool. These features aim to improve the user's ability to review, edit, and share OCR results.

## 2. Split-Pane UI Testing

### 2.1 Layout and Visibility
| Test Case ID | Description | Expected Result |
|--------------|-------------|-----------------|
| UI-01 | Side-by-Side View | Upon completion of OCR, the UI should show the original image and the Markdown editor side-by-side. |
| UI-02 | Responsive Layout | On mobile devices (width < 768px), the panes should stack vertically instead of being side-by-side. |
| UI-03 | Pane Resizing | If implemented, dragging the divider between panes should resize them proportionally. |
| UI-04 | Sync Scrolling | (Optional) Scrolling the Markdown editor should ideally keep the corresponding section of the image in view, and vice versa. |

### 2.2 Editor Functionality
| Test Case ID | Description | Expected Result |
|--------------|-------------|-----------------|
| UI-05 | Live Editing | Changes made in the Markdown editor should be preserved and reflected in the exported files. |
| UI-06 | Formatting Toolbar | If present, toolbar buttons (Bold, Italic, etc.) should correctly insert Markdown syntax. |

## 3. PDF/DOCX Export Testing

### 3.1 Export Actions
| Test Case ID | Description | Expected Result |
|--------------|-------------|-----------------|
| EXP-01 | DOCX Download | Clicking "Export to Word" (or similar) downloads a valid .docx file. |
| EXP-02 | PDF Download | Clicking "Export to PDF" (or similar) downloads a valid .pdf file. |
| EXP-03 | File Naming | Downloaded files should have meaningful names (e.g., `ocr_result_YYYYMMDD.docx`). |

### 3.2 Formatting Fidelity
| Test Case ID | Description | Expected Result |
|--------------|-------------|-----------------|
| EXP-04 | Heading Conversion | Markdown `#`, `##`, etc., should be converted to proper Word/PDF Heading styles. |
| EXP-05 | List Conversion | Bulleted and numbered lists should be correctly formatted as lists in the export. |
| EXP-06 | Inline Styles | **Bold** and *Italic* text in Markdown must be bold and italic in the exported document. |
| EXP-07 | UTF-8 Support | Special characters and accented letters must be rendered correctly in the export. |

## 4. Edge Cases & Stress Testing

| Test Case ID | Description | Expected Result |
|--------------|-------------|-----------------|
| EDGE-01 | Large Document | Exporting a result with >10,000 words should complete without timing out or crashing. |
| EDGE-02 | No Text | Attempting to export when no OCR text is present should be handled gracefully (e.g., disabled button or warning). |
| EDGE-03 | Complex Markdown | Test with tables, blockquotes, and nested lists to see how export handles complex structures. |

## 5. Environment & Tools
- **Browsers**: Chrome (latest), Firefox (latest), Safari (latest).
- **Office Tools**: Microsoft Word, LibreOffice, Adobe Acrobat Reader.
- **Testing Framework**: Manual verification + Playwright/Cypress for UI automation (future).
