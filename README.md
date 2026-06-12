# 🖼️ DescriptoImage – Text-to-Image Generation using GANs & Diffusion Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Status-Completed-green?style=flat" />
</p>

---

## 📌 Overview

DescriptoImage is a deep learning project that generates **fashion product images from text captions**.
It explores and compares multiple generative architectures — from simple GANs to Style GAN, Attention GAN,
and fine-tuned Stable Diffusion — trained on a real-world fashion dataset with image URLs and product title captions.

---

## ✨ Features

- 📝 Generates images from natural language fashion descriptions
- 🧠 Implements and compares **4 architectures**: Simple GAN, Style GAN, Attention GAN, and Stable Diffusion
- 🔤 Text encoding via **LSTM-based encoder** and **CLIP** (for diffusion)
- 📦 Dataset preprocessing: text cleaning, tokenization, padding, SMOTE class balancing
- 🖼️ Image preprocessing: URL-based downloading, normalization, resizing to 64×64 and 256×256
- 📊 Evaluated with **FID Score** and **CLIP Score**

---

## 🖼️ Generated Results

### 1️⃣ Simple GAN

| Epoch 50 | Epoch 600 |
|---|---|
| ![Simple GAN 50](simple%20gan%20-50%20epoch.png) | ![Simple GAN 600](simple%20gan-600%20epoch.png) |

---

### 2️⃣ Style GAN

| Epoch 50 | Epoch 600 |
|---|---|
| ![Style GAN 50](style%20gan-50%20epoch.png) | ![Style GAN 600](style%20gan%20-600%20epoch.png) |

---

### 3️⃣ Attention GAN

| Epoch 50 | Epoch 600 |
|---|---|
| ![Attention GAN 50](attention%20gan-50%20epoch.png) | ![Attention GAN 600](attention%20gan.png) |

---

### 4️⃣ Stable Diffusion (Fine-tuned)

![Stable Diffusion](stable%20diffusion%20-50%20epoch.png)

---

## 🏗️ Model Architectures

### 1️⃣ Simple GAN
```
Text Caption → LSTM Encoder ─┐
                              ├─► Concatenate → Generator → 64×64 Image
Random Noise ─────────────────┘
                        ↕
              Discriminator (Real/Fake)
```
- Trained for **50 and 600 epochs**
- Loss: Binary Crossentropy
- Optimizer: Adam (lr=1e-4)

---

### 2️⃣ Style GAN
```
Text Caption → LSTM Encoder ─┐
                              ├─► Style Modulation → Generator → 64×64 Image
Random Noise ─────────────────┘
                        ↕
              Discriminator (Real/Fake)
```
- Trained for **50 and 600 epochs**
- Improved image quality over Simple GAN

---

### 3️⃣ Attention GAN
```
Text Caption → LSTM Encoder → Attention Layer ─┐
                                                ├─► Generator → Image
Random Noise ──────────────────────────────────┘
```
- Attention mechanism focuses on relevant text features
- Better text-image alignment than basic GAN
- Trained for **50 and 600 epochs**

---

### 4️⃣ Stable Diffusion (Fine-tuned)
```
Caption → CLIP Tokenizer → CLIPTextModel → Encoder Hidden States
                                                    ↓
Image → VAE Encoder → Latents + Noise → UNet → Denoised Latents → VAE Decoder → Image
```
- Fine-tuned **U-Net** of `CompVis/stable-diffusion-v1-4`
- CLIP tokenizer: `openai/clip-vit-large-patch14`
- Trained with AdamW (lr=1e-4)

---

## 📂 Dataset

- **Source:** Fashion product dataset (Kaggle) with image URLs and product titles
- **Preprocessing:**
  - Combined `Usage` + `ProductTitle` → `CombinedTitle` as caption
  - Text cleaning: lowercasing, URL/punctuation removal
  - Tokenization with `num_words=5000`, padding to max sequence length
  - Images downloaded in parallel using `ThreadPoolExecutor`
  - Resized to `64×64` (GAN) and `256×256` (Diffusion)
  - SMOTE applied for class imbalance

---

## 🧪 Sample Test Captions Used

```
"ethnic Boys Printed Maroon Kurta"
"sports Men Gel Running White Sports Shoes"
"casual Men Aventura Sandal"
"casual Boys Expressions Yellow T-shirt"
"formal men brown shoes"
"casual girl greens shirt kidswear"
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras, PyTorch |
| Diffusion | Hugging Face Diffusers, Stable Diffusion v1.4 |
| Text Encoding | LSTM, CLIP (openai/clip-vit-large-patch14) |
| Data | Pandas, NumPy, SMOTE (imbalanced-learn) |
| Image Processing | OpenCV, PIL, Matplotlib |
| Evaluation | FID Score, CLIP Score |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Habibatariq24/DescriptoImage.git
cd DescriptoImage
```

### 2. Install dependencies
```bash
pip install tensorflow torch diffusers transformers accelerate
pip install datasets ftfy torchvision tqdm safetensors peft
pip install pandas numpy opencv-python pillow matplotlib scikit-learn imbalanced-learn
```

### 3. Run the notebook
```bash
jupyter notebook "text-to-image-generation uisng GANS.ipynb"
```

---

## 📁 Project Structure

```
DescriptoImage/
├── text-to-image-generation uisng GANS.ipynb  ← Main notebook
├── simple gan -50 epoch.png                   ← Simple GAN output (50 epochs)
├── simple gan-600 epoch.png                   ← Simple GAN output (600 epochs)
├── style gan-50 epoch.png                     ← Style GAN output (50 epochs)
├── style gan -600 epoch.png                   ← Style GAN output (600 epochs)
├── attention gan-50 epoch.png                 ← Attention GAN output (50 epochs)
├── attention gan.png                          ← Attention GAN output (600 epochs)
├── stable diffusion -50 epoch.png             ← Stable Diffusion output
└── README.md
```

---

## 📊 Evaluation Metrics

| Metric | Purpose |
|---|---|
| **FID Score** | Measures quality & diversity of generated images (lower = better) |
| **CLIP Score** | Measures text-image alignment (higher = better) |

---

## 🎓 Academic Context

- **Institution:** FAST NUCES, Lahore
- **Degree:** BS Data Science
- **Duration:** March – May 2025

---

## 👩‍💻 Author

**Habiba Tariq**  
[LinkedIn](https://www.linkedin.com/in/habiba-tariq24) • [GitHub](https://github.com/Habibatariq24) • habibatariq1248@gmail.com
