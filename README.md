# FFHQ Face Alignment

ffhq-align is a face alignment operation that allows gradient computation and runs on PyTorch GPU entirely.

This implementation is faster than original even running on CPU.

See original implementation which not support gradient computation and not run on GPU: [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)

## Install

run one of the following commands.

```bash
> pip install ffhq-align
```

```bash
> git clone https://github.com/kynk94/ffhq-align
> cd ffhq-align
> python setup.py install
```

## Examples

```txt
usage: align.py [-h] --input INPUT [--output OUTPUT] [--resolution RESOLUTION] [--batch_size BATCH_SIZE]

Face Alignment

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
  --output OUTPUT, -o OUTPUT
  --resolution RESOLUTION, -r RESOLUTION
  --batch_size BATCH_SIZE, -b BATCH_SIZE
```

```bash
> python align.py -i samples -r 512 -b 8
```

```python
import torch
from PIL import Image

from ffhq_align import Aligner

aligner = Aligner(padding_mode="blur")
image = Image.open("samples/input.jpg")

# Aligner returns tuple of aligned_image tensors.
aligned_image = aligner(image, resolution=512)[0]
```

```python
from typing import List, Optional

import numpy as np
import torch
from numpy import ndarray
from PIL import Image
from torchvision import transforms

from ffhq_align import LandmarkFA, image_align

landmark_model = LandmarkFA()

image = Image.open("samples/input.jpg")
transform =  transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)
tensor = transform(image)[None]  # (1, 3, H, W)

# landmark_model returns batch landmarks.
# if there is no faces, the value is None.
landmarks: List[Optional[ndarray]] = landmark_model(tensor)
if any(l is None for l in landmarks):
    raise ValueError("Face not Found")

landmarks = torch.from_numpy(np.array(landmarks))  # (1, 68, 2)
aligned_image = image_align(tensor, landmarks, resolution=512)  # (1, 3, 512, 512)
```
