# Sparse-View 3D Gaussian Splatting via Local Correlation Priors and 3D Spatial Pruning

This repository contains the official implementation of the final project for the "Deep Learning for 3D Computer Vision" course. The project tackles the severe degradation of 3D Gaussian Splatting (3DGS) under sparse-view conditions (e.g., 3-4 images with wide baselines). 

By replacing error-prone absolute depth alignments with scale-invariant **Local Pearson Correlation (LPC)** priors, and integrating active 3D spatial pruning (**Gradient-Driven Natural Selection - GNS**), we present a hybrid architecture that successfully suppresses floating artifacts and stabilizes the underlying geometric topology without compromising memory efficiency.

---

## 1. Originality & Code Attribution

This project builds upon the foundational flattened-rendering architecture of **SMDGS** (Scale-Aligned Monocular Depth-Guided 3D Gaussian Splatting) and **Vanilla 3DGS**. 

To strictly adhere to the academic requirements regarding code attribution, the distinction between the baseline frameworks and our explicit contributions is detailed below:

*   **External Baselines (Not written by us):** The core rasterization engines (`diff-gaussian-rasterization`, `diff-plane-rasterization`), SfM alignment loaders, and the initial training structure were adopted from the SMDGS framework and the original 3DGS repository.

*   **Monocular Depth Estimator:** We utilized the pre-trained `DepthAnythingV2` (ViT-B) model for offline depth prior generation. The model's weights and base architecture are external.

*   **Our Core Algorithmic Contributions (`train.py`):** We heavily modified the training loop to conduct our ablation study. Our additions (marked with `MODIFIED` comments in the code) include:

    *   **Local Pearson Correlation (LPC):** A highly memory-efficient, analytical implementation of local depth regularization (`patch_pearson_correlation_loss`) to overcome global scale/shift ambiguity.

    *   **3D Spatial Pruning (GNS):** The integration of a Gradient-Driven Natural Selection mechanism, carefully engineered with a time-dependent "Recovery Window" to prevent architectural collisions with the native 3DGS opacity resets.

    *   **Virtual Pseudo-Views (Extension A):** An alternating optimization strategy applying Total Variation (TV) loss on unobserved virtual trajectories to enforce spatial smoothness.

*   **Our Engineering and Automation Contributions (Written from scratch):**

    *   `setup_env.sh` & `requirements.txt`: Robust, automated environment setup scripts.

    *   `project_config.py`: A centralized global configuration file handling dynamic paths and pipeline logic.

    *   `run_pipeline.sh`: The master execution script orchestrating the entire automated pipeline (Pre-processing -> Baselines -> 4 Ablation Configurations).

    *   `preprocessing/filter_colmap.py` & `preprocessing/prepare_depth_priors.py`: Custom scripts to filter SfM data for sparse settings and generate offline depth priors using DepthAnythingV2.

---

## 2. Hardware Requirements & Constraints

This entire pipeline was rigorously optimized to run on highly constrained consumer hardware, specifically an **NVIDIA RTX 4050 GPU limited to 6GB of VRAM**.
To achieve this, the default configuration (`project_config.py`) explicitly sets `RESOLUTION_SCALE = 2` (half resolution) and enforces offline depth generation to prevent memory saturation during the intense backpropagation of geometric constraints.

---

## 3. Data Preparation & Weights

To ensure seamless reproducibility, we have included a minimal reproducible dataset directly within the repository. However, you also have the option to build the sparse-view setup from scratch using other scenes.

**Option A: Using the Provided Minimal Dataset (Recommended)**

The `/data/bicycle` and `/data/garden` folders already contains the 4 carefully selected sparse-view images and the correspondingly filtered COLMAP files (`cameras.bin`, `images.bin`, `points3D.bin`). The `project_config.py` is pre-configured to use this scene by default, meaning you can skip to Section 4 immediately.

**Option B: Custom Dataset Extraction (From Scratch)**

If you wish to run the pipeline on a different scene, you must acquire the dataset and select the sparse views manually:

1. **Download Mip-NeRF 360 Dataset:** Download the original dataset (e.g., the `bicycle` or `garden` scene) which contains hundreds of high-resolution images and pre-calculated COLMAP data.

2. **Select Sparse Views:** Choose exactly 4 images from the original `images` folder that provide a reasonable overlap around the target object.

3. **Place the Data:** Ensure your project folder structure matches the expected pipeline input. Place the 4 selected images and the corresponding ORIGINAL COLMAP files into the `data` directory as follows:
    ```text
    Sparse_View_3DGS/
    └── data/
        └── <your_custom_scene>/
            ├── images/            <-- Place your 4 selected images here
            └── sparse/
                └── 0/             <-- Place the original cameras.bin, images.bin, points3D.bin here
    ```

4. **Configure Scene Name:** Open `project_config.py` and update the `SCENE_NAME` variable to match your new dataset folder (e.g., `SCENE_NAME = "your_custom_scene"`).

**Depth Model Weights**

The automated setup script (see Section 4) will automatically download the required `DepthAnythingV2` (ViT-B) weights directly into the `preprocessing/depth_estimation/weights/` directory. If you bypass the setup script, you must manually download and place the `depth_anything_v2_vitb.pth` file in that directory before running the pipeline.

---

## 4. Environment Setup (One-Click Install)

Given the complexity of compiling C++/CUDA submodules for 3DGS, we provide a unified bash script to build the entire Conda environment flawlessly.

1. Open your WSL/Linux terminal in the project root directory.
   
2. Grant execution permissions to the script:
   ```bash
   chmod +x setup_env.sh
   ```

3. Run the setup script:
   ```bash
   ./setup_env.sh
   ```
   **What this script does automatically:**

   - Creates a Conda environment named `sparse_3dgs` (Python 3.10).

   - Installs the CUDA 12.1 Toolkit directly via Conda (avoiding OS-level key issues).

   - Installs the exact matching PyTorch version and compiles `pytorch3d` from the source.

   - Installs all required standard Python libraries via `requirements.txt`.
  
   - Compiles all internal C++/CUDA submodules (rasterizers and KNN) for both the Vanilla 3DGS and SMDGS branches.

4. Once completed, activate the environment:
   ```bash
   conda activate sparse_3dgs
   ```

---

## 5. Reproducing the Results (Automated Pipeline)

To reproduce the exact ablation study and results presented in our final report, we provide an automated master script.

1. Grant execution permissions:
   ```bash
   chmod +x run_pipeline_2.sh
   ```

2. Execute the pipeline:
   ```bash
   ./run_pipeline.sh
   ```

   **Pipeline Execution Flow:**

   The script performs the following operations sequentially without manual intervention:
   
   1. **Pre-processing (`filter_colmap.py`):** Parses the original COLMAP files and strips away unobserved cameras/points, generating a tailored `.bin` subset exclusively for your 4 chosen images.

   2. **Depth Priors (`prepare_depth_priors.py`):** Loads the DepthAnythingV2 model, infers monocular depth for the 4 images, and strictly formats them as Numpy arrays (`.npy`) expected by the SMDGS engine

   3. **Baseline Training:** Trains the unmodified Vanilla 3DGS and the SMDGS baselines.

   4. **Ablation Study (Runs 16, 17, 18, 20):** Executes our custom configurations by selectively passing flags to `train.py`:
       - `[Run 16]`: Hybrid LPC + Empty Space + Dropout (`--use_lpc`).

       - `[Run 17]`: Run 16 (Hybrid LPC + Empty Space) replacing Dropout with 3D Pruning GNS (`--use_lpc --use_gns`).

       - `[Run 18]`: Run 17 (Hybrid LPC + Empty Space + GNS) + Extension A TV Loss (`--use_lpc --use_gns --use_extension_a --lambda_tv 0.1`).

       - `[Run 20]`: Run 16 (Hybrid LPC + Empty Space + Dropout) + Extension A without GNS (`--use_lpc --use_extension_a --lambda_tv 0.1`).

    5. **Evaluation:** Automatically calls `render.py` and `metrics.py` after each run, outputting the quantitative scores (PSNR, SSIM, LPIPS) into a `results.json` file inside each respective output folder

**Data Flow and Output Paths:**

*   All intermediate processing files (e.g., generated `.npy` depth maps) are managed securely within the `data/<scene_name>/` directory.

*   All training checkpoints, point clouds (`.ply`), rendered images, and the final `results.json` metric files are saved automatically to: `output/<scene_name>/<run_name>/`.

---

## 6. Advanced Usage (Manual Execution)

If you wish to test specific hypotheses or bypass the automated pipeline, you can manually trigger `train.py` using our custom ablation flags.

Example manual execution for the hybrid GNS model:
```bash
python train.py \ 
    -s data/bicycle/colmap \
    --resolution 2 \
    --iterations 15000 \
    --model_path output/bicycle/manual_run \
    --use_lpc \
    --use_gns
```

**Available Research Flags:**

- `--use_lpc`: Activates the Local Pearson Correlation depth regularization and empty-space penalty.

- `--use_gns`: Activates Gradient-Driven Natural Selection (disables baseline opacity dropout).

- `--use_extension_a`: Enables Unseen-Viewpoint Spatial Smoothing (Pseudo-views).

- `--lambda_tv <float>`: Sets the weight for the Extension A Total Variation loss (default: 0.1).

*Note: When activating `--use_extension_a`, we recommend extending the training schedule by passing both `--iterations 16000` and `--geom_prior_until_iter 16000` to compensate for the alternating optimization steps dedicated to the pseudo-views.*

---

## Acknowledgments
This project was developed as the final research assignment for the **Deep Learning for 3D Computer Vision** course. 
*   **Author:** David Israel Naki
*   **Course Instructor:** Sagie Benaim