# data/

Not versioned in git. Expected layout (YOLO format) — see `docs/dataset.md`:

```
data/
  jass.yaml          # dataset config (versioned)
  raw/               # original photos, untouched
  jass/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
```
