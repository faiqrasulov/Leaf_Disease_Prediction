# Plant leaf disease classification with fastai

> Course assignment from 2023, kept as it was written. The notes under "Reading the result" were added in November 2026.

Classifies leaf photos into 38 species-and-disease classes (the augmented PlantVillage set from
Kaggle, `vipoooool/new-plant-diseases-dataset`) with a pretrained ResNet-34 in fastai.

## Method

Random-resized crops to 128 px, 15% of the images held out for validation, learning rate from
`lr_find` (2.5e-3), 5 epochs of one-cycle training on a Colab GPU, under 2 minutes per epoch.

## Result

| Epoch | Validation accuracy |
|---|---|
| 1 | 95.5% |
| 3 | 98.8% |
| 5 | **99.4%** |

The most frequent confusion is between two corn diseases: grey leaf spot predicted as northern
leaf blight 16 times, and the reverse 7 times.

## Reading the result

- **99.4% is optimistic.** The Kaggle set was augmented offline, so rotated and flipped copies of
  one photo exist as separate files. The notebook splits the files at random, which puts copies of
  the same photo on both sides. A split by original photo is the honest check.
- PlantVillage photos are single leaves on a plain background. Models trained on it lose much of
  their accuracy on photos taken in the field.

## Files

| File | Content |
|---|---|
| `Leaf_Disease_Fastai.ipynb` | Notebook (written for Google Colab, fastai 2.6.3) |
| `About.txt` | Original brief |

The images (2.7 GB) are downloaded from Kaggle by the notebook.
