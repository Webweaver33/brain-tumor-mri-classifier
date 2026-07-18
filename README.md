# Brain Tumor MRI Classifier — Streamlit Dashboard

## Folder structure
```
brain-tumor-classifier/
├── app.py                  # Streamlit dashboard (run this)
├── utils.py                # Model loading, prediction, Grad-CAM logic
├── requirements.txt
├── models/                 # <-- put your trained model files here
│   ├── vit-base-brain-tumor/   (folder — from vit_model.save_pretrained() in Colab)
│   ├── vit_base_best.pth
│   └── resnet50_best.pth
└── sample_images/          # optional: a few MRI images to demo with
```

## Where the dataset goes
You do NOT need the full dataset zip to run the dashboard — the dashboard only
needs the **already-trained model files** above. Keep training in Colab (GPU),
then copy just those 3 items into `models/`.

If you also want to keep the dataset around locally (e.g. to grab test images),
unzip it to a `data/` folder in this project — it's not required by app.py.

## Getting the trained model files out of Colab
At the end of your Colab training run, add this cell to zip and download them:
```python
import shutil
from google.colab import files

shutil.make_archive('/content/trained_models', 'zip', root_dir='/content', base_dir='.')
# Or, more targeted:
import zipfile
with zipfile.ZipFile('/content/models_for_streamlit.zip', 'w') as zf:
    zf.write('/content/vit_base_best.pth', 'vit_base_best.pth')
    zf.write('/content/resnet50_best.pth', 'resnet50_best.pth')
    for root, dirs, filenames in os.walk('/content/vit-base-brain-tumor'):
        for fname in filenames:
            fpath = os.path.join(root, fname)
            arcname = os.path.join('vit-base-brain-tumor', os.path.relpath(fpath, '/content/vit-base-brain-tumor'))
            zf.write(fpath, arcname)

files.download('/content/models_for_streamlit.zip')
```
Then unzip its contents directly into your local `models/` folder.
