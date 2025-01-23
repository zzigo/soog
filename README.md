<div align="center">
  <img src="frontend/public/android-chrome-512x512.png" alt="SOOG Logo" width="200"/>
</div>

# SOOG - Speculative Organology Organogram Generator

![Nuxt](https://img.shields.io/badge/Nuxt-35495E?style=for-the-badge&logo=nuxt.js&logoColor=4FC08D)
![Bun](https://img.shields.io/badge/Bun-000000?style=for-the-badge&logo=bun&logoColor=FFFFFF)
![Vue](https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vue.js&logoColor=4FC08D)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=FFFFFF)
![ChatGPT API](https://img.shields.io/badge/ChatGPT%20API-412991?style=for-the-badge&logo=openai&logoColor=FFFFFF)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=FFFFFF)
![Matplotlib](https://img.shields.io/badge/Matplotlib-013243?style=for-the-badge&logo=python&logoColor=FFFFFF)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=FFFFFF)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=FFFFFF)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=FFFFFF)
![MathBERT](https://img.shields.io/badge/MathBERT-412991?style=for-the-badge&logo=bert&logoColor=FFFFFF)
[![GitHub stars](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social)](https://github.com/huggingface/transformers)
[![GitHub stars](https://img.shields.io/github/stars/pytorch/pytorch.svg?style=social)](https://github.com/pytorch/pytorch)
[![GitHub stars](https://img.shields.io/github/stars/numpy/numpy.svg?style=social)](https://github.com/numpy/numpy)
[![GitHub stars](https://img.shields.io/github/stars/matplotlib/matplotlib.svg?style=social)](https://github.com/matplotlib/matplotlib)
[![GitHub stars](https://img.shields.io/github/stars/pallets/flask.svg?style=social)](https://github.com/pallets/flask)
[![GitHub stars](https://img.shields.io/github/stars/psf/requests.svg?style=social)](https://github.com/psf/requests)
[![GitHub stars](https://img.shields.io/github/stars/openai/openai-python.svg?style=social)](https://github.com/openai/openai-python)
[![GitHub stars](https://img.shields.io/github/stars/tqdm/tqdm.svg?style=social)](https://github.com/tqdm/tqdm)
[![GitHub stars](https://img.shields.io/github/stars/NVIDIA/nccl.svg?style=social)](https://github.com/NVIDIA/nccl)
[![GitHub stars](https://img.shields.io/github/stars/huggingface/huggingface_hub.svg?style=social)](https://github.com/huggingface/huggingface_hub)

SOOG is an innovative web application that helps visualize and create musical instruments using the organogram technique, originally developed by ethnomusicologist Mantle Hood. The application extends this methodology to enable speculative instrument design through geometric and acoustic manipulation.

## Core Features

1. **Visualization System**

   - Abstract instrument representation using geometric shapes
   - Color-coded material identification
   - Movement and interaction indication through arrows
   - Acoustic space visualization
   - Measurable component representation

2. **Instrument Classification**

   - Idiophones: squares
   - Membranophones: horizontal rectangles
   - Chordophones: vertical rectangles
   - Aerophones: circles
   - Electronophones: rhombus

3. **Material Color Coding**
   - Wood: orange
   - Bamboo: yellow
   - Skin: pink
   - Glass: green
   - Stone: white
   - Water: blue
   - And more...

## Technical Implementation

### Frontend Stack

1. **Core Technologies**

   - Nuxt 3: Vue-based framework for server-side rendering
   - Vite: Next-generation frontend tooling
   - Bun: Fast JavaScript runtime and package manager
   - TypeScript: Type-safe development

2. **UI Components**
   - Vue.js 3 with Composition API
   - Three.js for 3D visualization
   - D3.js/Plotly for vector graphics
   - WebGL for advanced rendering

### Backend Architecture

1. **AI and ML Integration**

   - GPT-4 API for natural language processing
   - Custom BERT models fine-tuned for:
     - Instrument classification
     - Material recognition
     - Acoustic property prediction
   - Multi-modal model combining text and image analysis

2. **Redis Caching System**

   - GPT Response Caching (1 hour TTL)
   - Plot Image Caching (30 minutes TTL)
   - Model Caching (24 hours TTL)
   - Static Content Caching (12 hours TTL)

3. **Cache Components**
   - `cache_config.py`: Redis settings and TTLs
   - `cache_manager.py`: Redis operations
   - `cache_decorators.py`: Caching decorators

### Model Training

1. **BERT-based Models**

   - Base model: DistilBERT
   - Training data:
     - Academic papers on organology
     - Instrument specifications
     - Material properties
   - Fine-tuning objectives:
     - Text classification
     - Feature extraction
     - Relationship mapping

2. **Multi-modal Integration**
   - CLIP for image-text alignment
   - Custom attention mechanisms
   - Cross-modal feature fusion

## Roadmap

### Current Implementation

- Basic organogram generation
- Material and shape visualization
- Acoustic space representation
- Interactive UI with keyboard shortcuts

### Planned Features

1. **Query Parser Enhancement**

   - Natural language processing for input classification
   - Keyword-based modality mapping:
     - Text responses
     - Vector visualization (matplotlib, d3, plotly)
     - 3D space (three.js)

2. **Advanced Training Systems**

   - Skill-Based Model Training:
     - Natural language generation
     - Code generation for visualizations
     - Logic-based reasoning
   - Dynamic Prompt Engineering:
     - Context-aware prompt generation
     - Specialized expertise activation

3. **Input Parsing Categories**
   - Draw: Vector-based visualization
   - Simulate: 3D space representation
   - Explain: Text-based response

### Future Development

1. **Extended Organological Features**

   - Acoustic simulation integration
   - Material physics modeling
   - Interactive sound generation
   - Real-time collaboration tools

2. **AI Enhancement**

   - Improved instrument recognition
   - Advanced morphological analysis
   - Historical instrument database integration
   - Cross-cultural instrument mapping

3. **Research Tools**
   - Academic citation integration
   - Research paper generation
   - Comparative analysis tools
   - Dataset visualization

## Setup Requirements

1. Install Redis Server:

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu
sudo apt-get install redis-server
sudo systemctl start redis-server
```

2. Install Dependencies:

```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash

# Install project dependencies
bun install        # Frontend
pip install -r requirements.txt  # Backend
```

## Academic Context

SOOG is part of a doctoral research project in Music Performance Research at the Hochschule der Künste Bern, extending Hood's organogram methodology into the domain of speculative organology.

## References

Hood, Mantle (1982). _The ethnomusicologist_ (2nd ed.). Kent State University Press.
