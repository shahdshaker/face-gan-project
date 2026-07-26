# Realistic Human Face Generation using GAN

## Objective
This project implements a Generative Adversarial Network (GAN) to generate
realistic synthetic human face images from random noise vectors. It covers
the full pipeline: dataset preparation, model architecture design, adversarial
training, and evaluation of generated image quality over time.

## Dataset
- **Source**: CelebA (Large-scale CelebFaces Attributes Dataset)
- **Size used**: ~202,599 aligned face images
- **Preprocessing**:
  - Center-cropped each image to a square region (178x178) to remove
    background and hair edges
  - Resized to 64x64 pixels
  - Normalized pixel values from [0, 255] to [-1, 1] to match the
    Generator's `tanh` output activation

## Model Architecture (DCGAN)

**Generator** — maps a 100-dimensional random latent vector to a 64x64x3 RGB image:
- Dense layer projecting to an 8x8x256 feature map
- 3x Conv2DTranspose blocks (BatchNorm + LeakyReLU) progressively upsampling
  8x8 → 16x16 → 32x32 → 64x64
- Final Conv2DTranspose layer with `tanh` activation producing the RGB output

**Discriminator** — classifies a 64x64x3 image as real or fake:
- 3x Conv2D blocks (LeakyReLU + Dropout) downsampling the image
- Flatten → Dense(1) producing a single real/fake logit

**Loss function**: Standard binary cross-entropy adversarial loss
(non-saturating generator loss)
**Optimizer**: Adam (learning rate = 1e-4, beta_1 = 0.5) for both networks

## Training
- **Platform**: Google Colab (T4 GPU)
- **Epochs**: 50
- **Batch size**: 128
- A fixed set of 16 latent noise vectors was used to generate a sample
  image grid after every epoch, so the same points in latent space could
  be visually tracked as they evolved into sharper faces over time.

## Evaluation Approach
GAN loss values alone are not a reliable indicator of image quality — a
"good" discriminator loss doesn't necessarily mean realistic output.
The primary evaluation method used here is **visual inspection** of the
generated sample grids saved at each epoch, supported by the loss curves
for tracking training dynamics.

## Results

**Loss curves** (`results/loss_curve.png`):
The Discriminator loss stayed low and relatively stable (~0.3–0.6)
throughout training, while the Generator loss remained higher and more
volatile (~2.5–3.0). This indicates the Discriminator was consistently
outperforming the Generator — a common dynamic in GAN training — meaning
the Generator struggled to fully fool the Discriminator within 50 epochs.

**Generated faces over time** (`results/epoch_comparison.png`):
- **Early epochs (0–5)**: Output was largely unstructured noise/color blotches.
- **Mid training (~epoch 15–30)**: General face-like shapes and skin-tone
  regions began to emerge, though details remained blurry.
- **Final epochs (~epoch 49)**: More defined facial structure is visible,
  though outputs are still lower-fidelity than a fully converged model —
  additional training epochs would likely continue to improve realism.

## Limitations
- Training was limited to 50 epochs due to compute/time constraints; DCGANs
  typically benefit from significantly more epochs (100+) for higher-fidelity
  results.
- The imbalance between Generator and Discriminator loss suggests the
  Discriminator became too strong relative to the Generator — potential
  improvements include reducing the Discriminator's learning rate relative
  to the Generator's, or adding label smoothing.
- Output resolution was kept at 64x64 for training efficiency; higher
  resolutions (128x128) would require a deeper architecture and more
  training time.

## Project Structure
face-gan-project/
├── Face_GAN_Colab.ipynb # Full training notebook (Colab, GPU)
├── README.md
└── results/
├── loss_curve.png
├── epoch_comparison.png
└── samples/
├── epoch_0000.png
├── epoch_0015.png
├── epoch_0030.png
└── epoch_0049.png
## How to Run
1. Upload `img_align_celeba` (CelebA dataset) to Google Drive.
2. Open `Face_GAN_Colab.ipynb` in Google Colab.
3. Set Runtime → Change runtime type → GPU (T4).
4. Run all cells in order. Results (samples, checkpoints, loss curves) are
   saved automatically to Google Drive under `face_gan_results/`.

## Model Weights
Trained generator/discriminator weights (`.h5` files) are not included in
this repository due to file size. They are available here: **[add your
Google Drive link if you want to share them]**