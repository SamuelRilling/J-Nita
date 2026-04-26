import os
import sys

# Add the parent directory to the path so we can import ocr_workflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_workflow import OCRWorkflow

def test_docx_export():
    print("Testing DOCX export...")
    
    # Initialize workflow (minimal params)
    workflow = OCRWorkflow(input_folder="tests/dataset", handwriting_mode=True)
    
    test_text = """# Test Document
This is a test of the **J-Nita** DOCX export functionality.

## Features
- Preserves **Bold** text
- Preserves *Italic* text
- Handles `Inline Code`
- Supports lists:
  1. Item one
  2. Item two

### Nested Section
More text here to verify paragraph breaks.
"""
    
    output_file = "tests/test_output.docx"
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print(f"Exporting to {output_file}...")
    try:
        workflow._save_as_docx(test_text, output_file)
        if os.path.exists(output_file):
            print(f"✓ Success: {output_file} created. Size: {os.path.getsize(output_file)} bytes")
        else:
            print("✗ Failure: Output file not created.")
    except Exception as e:
        print(f"✗ Error during export: {e}")

if __name__ == "__main__":
    test_docx_export()
