from PIL import Image, ImageDraw
import io
from django.core.files.base import ContentFile

def dummy_ai_module(image_file):
    original = Image.open(image_file)
    
    processed = original.copy()
    vascular = original.copy()
    total_root = original.copy()
    xylem = original.copy()
    
    def add_text(img, text):
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, fill=(255, 0, 0))
        return img

    processed = add_text(processed, "Processed")
    vascular = add_text(vascular, "Vascular Bundle Mask")
    total_root = add_text(total_root, "Total Root Mask")
    xylem = add_text(xylem, "Xylem Mask")
    
    vascular_area = 123.4
    vascular_diameter = 45.6
    xylem_count = 7
    xylem_diameter = 8.9
    xylem_details = [
        {"id": i+1, "area": 10+i, "diameter": 2.5+i} for i in range(xylem_count)
    ]
    
    def image_to_content(img, name):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return ContentFile(buffer.getvalue(), name=name)
    
    result = {
        "processed_file": image_to_content(processed, 'processed.jpg'),
        "vascular_file": image_to_content(vascular, 'vascular.jpg'),
        "total_root_file": image_to_content(total_root, 'total_root.jpg'),
        "xylem_file": image_to_content(xylem, 'xylem.jpg'),
        "vascular_area": vascular_area,
        "vascular_diameter": vascular_diameter,
        "xylem_count": xylem_count,
        "xylem_diameter": xylem_diameter,
        "xylem_details": xylem_details,
    }
    return result