# Semantic-Driven Image-to-Prompt Inversion via Vision-Language Models and LLM Refinement

An end-to-end multi-modal framework designed for **Prompt Inversion Analysis**, implemented entirely inside a Jupyter Notebook environment.

The system automatically extracts semantic baselines from target images using vision-language models, leverages Large Language Models (LLMs) to optimize and mutate text prompts for Latent Consistency Models (LCM), and systematically evaluates synthesized outputs using multi-dimensional perceptual and semantic metrics.

---

## ⚙️ System Requirements & Dependencies

The pipeline requires:

* Python **3.10+**
* A hardware accelerator (**CUDA-compatible GPU strongly recommended**)
* Jupyter Notebook, JupyterLab, VS Code, or Google Colab

---

## 🔑 API Credentials Setup

Before running the notebook, provide your Google GenAI API credentials.

### Option 1 — Environment Variable

Set the API key before launching Jupyter:

```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

### Option 2 — Direct Initialization

Populate the API key inside the notebook initialization cell:

```python
from google import genai

client = genai.Client(
    api_key="YOUR_API_KEY_HERE"
)
```

---

## 📦 Dependency Installation

All required dependencies are automatically installed through inline `!pip` commands inside the notebook.

For local execution, install the full stack manually.

### Install PyTorch with CUDA Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Install Core Libraries

```bash
pip install -q -U google.genai transformers pillow pandas matplotlib seaborn
```

---

## 🚀 Execution Instructions

### Running the Notebook

1. Open the `.ipynb` notebook using:

   * Google Colab
   * JupyterLab
   * VS Code

2. Ensure the runtime is connected to a GPU accelerator:

   * NVIDIA T4
   * V100
   * A100
   * Equivalent CUDA-enabled device

3. Execute the notebook sequentially from top to bottom (`Shift + Enter`).

---

## 🔄 Notebook Execution Phases

The notebook is organized into four main analytical phases.

### Phase 0 — Environment Initialization

* Installs the Google GenAI SDK
* Loads all required Python packages
* Verifies runtime configuration

---

### Phase 1 — Visual Feature Extraction

* Loads target images
* Extracts semantic descriptions using:

  * BLIP
  * Florence-2
  * Additional vision-language models (optional)

The generated captions serve as the initial semantic baseline for prompt inversion.

---

### Phase 2 — Space Exploration & Prompt Expansion

* Connects to **Gemini 2.5 Flash**
* Optimizes extracted captions
* Generates prompt mutations and variations
* Expands the semantic search space for downstream image synthesis

---

### Phase 3 — Quantitative Screening

* Executes automated batch generation
* Produces candidate images via the diffusion model
* Computes similarity and distance metrics
* Builds ranking and evaluation matrices

---

### Phase 4 — Statistical Data Profiling

* Aggregates experimental results using Pandas
* Generates statistical summaries
* Produces publication-ready visualizations using Seaborn and Matplotlib
* Identifies Pareto-optimal prompt candidates through multi-objective analysis

---

## 📊 Evaluation Metrics

The framework evaluates synthesized outputs across three complementary scientific dimensions.

| Metric              | Evaluation Scope                                 | Optimization Objective |
| ------------------- | ------------------------------------------------ | ---------------------- |
| **CLIP Similarity** | Semantic alignment with the target image/context | **Maximize ↑**         |
| **LPIPS Distance**  | Human perceptual similarity                      | **Minimize ↓**         |
| **Pixel RMSE**      | Pixel-level reconstruction error                 | **Minimize ↓**         |

### Objective Summary

* Higher **CLIP Similarity** indicates stronger semantic consistency.
* Lower **LPIPS** indicates greater perceptual resemblance.
* Lower **RMSE** indicates improved structural reconstruction fidelity.

---

## 📈 Generated Outputs

During execution, the framework automatically creates a dedicated output directory containing:

* Generated images
* Intermediate prompt candidates
* Statistical summaries
* Similarity matrices
* Distance matrices
* Pareto frontier visualizations
* Experiment metadata

Supported output formats include:

```text
.csv
.json
.pdf
.png
```

---

## 📂 Output Structure

Example:

```text
outputs/
│
├── generated_images/
├── prompts/
├── metrics/
├── reports/
├── plots/
│
├── results.csv
├── metrics.json
└── pareto_frontier.pdf
```

---

## 🎯 Research Objective

The primary goal of this framework is to investigate **Prompt Inversion** by combining:

* Vision-language semantic extraction
* LLM-driven prompt optimization
* Diffusion-based image synthesis
* Multi-objective quantitative evaluation

This enables systematic exploration of the relationship between image content and the textual prompts capable of reproducing it with high semantic and perceptual fidelity.
