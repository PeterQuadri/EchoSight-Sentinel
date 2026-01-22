import zipfile
import xml.etree.ElementTree as ET
import os
import re

pptx_path = r"d:\DOCUMENTS\RAIN\AIML\Second_semester\Project\EchoSight Sentinel by Peter Quadri.pptx"

def get_slide_text(xml_content):
    root = ET.fromstring(xml_content)
    text_chunks = []
    for elem in root.iter():
        if elem.tag.endswith('}t'):
            if elem.text:
                text_chunks.append(elem.text)
    return " ".join(text_chunks)

if not os.path.exists(pptx_path):
    print(f"PPTX not found at {pptx_path}")
    exit(1)

try:
    with zipfile.ZipFile(pptx_path, 'r') as z:
        slides = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
        
        def get_slide_number(name):
            match = re.search(r'slide(\d+)\.xml', name)
            return int(match.group(1)) if match else 0
        
        slides.sort(key=get_slide_number)
        
        with open("pptx_content.txt", "w", encoding="utf-8") as f:
            for slide in slides:
                f.write(f"--- SLIDE {get_slide_number(slide)} ---\n")
                content = z.read(slide)
                text = get_slide_text(content)
                f.write(text + "\n\n")
            print("Extraction complete. Check pptx_content.txt")

except Exception as e:
    print(f"Error: {e}")
