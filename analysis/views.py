from django.shortcuts import render
from .forms import ImageUploadForm
from .dummy_ai import dummy_ai_module
from .models import Analysis
from PIL import Image

def upload_images(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        print("FILES:", request.FILES)
        print("Form valid?", form.is_valid())
        if form.is_valid():
            images = form.cleaned_data['images']
            results = []
            for img in images:
                try:
                    Image.open(img)
                except Exception as e:
                    print(f"File {img.name} is not a valid image: {e}")
                    continue

                ai_result = dummy_ai_module(img)
                
                processed_instance = Analysis(
                    original_image='path/to/original.jpg',
                    processed_image='path/to/processed.jpg',
                    vascular_bundle_image='path/to/vascular.jpg',
                    total_root_image='path/to/total_root.jpg',
                    xylem_image='path/to/xylem.jpg',
                    vascular_area=ai_result["vascular_area"],
                    vascular_diameter=ai_result["vascular_diameter"],
                    xylem_count=ai_result["xylem_count"],
                    xylem_diameter=ai_result["xylem_diameter"],
                    xylem_details=ai_result["xylem_details"]
                )
                processed_instance.save()
                results.append(processed_instance)
            return render(request, 'results.html', {'results': results})
        else:
            print(form.errors)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
