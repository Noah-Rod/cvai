# data/

Card photos (`raw-cards/`, `cards/`) are in git. Generated data (`synth/`, `exports/`)
is git-ignored: it is reproducible from the cards and a seed. Layout (see docs/dataset.md):

```
data/
  raw-cards/image_<code>.jpg   # 36 original card photos (input of tools/extract_cards.py)
  cards/<label>.png            # 36 straightened card faces (tools/extract_cards.py)
  backgrounds/*.jpg            # optional: background photos for the generator
  hands/*.png                  # optional: RGBA cut-outs of thumbs/hands
  synth/<name>/{train,val}/    # generator output: images/ + annotations.jsonl
  real/train/                  # real labeled photos (Label Studio → annotations.jsonl)
  real/test/images/ + hands.csv  # real test set: card list per photo
  exports/<name>/data.yaml     # YOLO datasets per box variant (jass_cv.dataset export)
```
