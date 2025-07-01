# Real-time-6D-Pose-Estimation-with-On-the-Fly-Object-Reconstruction
## 0) Clone the repository in your system

> **Note:** Make sure to read the requirements to running this pipeline. (requirements are at the end of the README)

```bash
git clone https://github.com/Yugdes/Real-time-6D-Pose-Estimation-with-On-the-Fly-Object-Reconstruction.git
```

## 1) Data Download

### Download Pretrained Weights

#### Segmentation Network (XMem)
- Download: [`XMem-s012.pth`]([https://example.com/path/to/XMem-s012.pth](https://drive.google.com/file/d/1LJ6U3NmI9MoUKG27mzqlgP1ioHWq-a5e/view))
- Place the file in: `.Object_Reconstruction/BundleSDF/BundleTrack/XMem/saves/`

#### Feature Matching Network (LoFTR)
- Download: [`outdoor_ds.ckpt`]([https://example.com/path/to/outdoor_ds.ckpt](https://drive.google.com/drive/folders/11b1-Wzxcn7LpmTgHPqlC3H1ZzGsB6j6R))
- Place the file in: `.Object_Reconstruction/BundleSDF/BundleTrack/LoFTR/weights/`

## 1.1) (Optional) Conda environment setup for running data capture script
```bash
conda create -n obj_recon python=3.10
pip install -r requirements.txt
```
## 2) Docker Setup

To set up the Docker environment, run the following command:

```bash
docker build --network host -t nvcr.io/nvidian/bundlesdf .
```
## 3) Capturing the data (To be done outside docker container)
The data is captured on Intel RealSense D435I.

  ### why this device?
  - it provides synchronized depth, RGB, and IMU (accelerometer + gyroscope) data, enabling accurate 3D reconstruction and camera pose estimation. Its wide field of view and high-resolution depth sensing make it ideal for dense mapping and visual-inertial odometry tasks.
> **Note:** Make sure to use the same device for both recinstruction and tracking.

To capture the data
   ```bash
   conda activate obj_recon        # if you have created a conda environment 
   cd Object_Reconstruction
   ./capture.sh
   ```
Enter the name of the object you want to reconstruct.

After then a gui will pop up showing you the recording from the device, try to show the objects from all angles into the device with less hand involvement for accurate reconstruction.

Once you have shown the object from all the angles press Enter to stop the recording.

After the recording is over, a masking window will pop up where you have to manually mask the image shown on the gui by clicking on the points around the boundary of the object as shown in the image.

![Demo Image](./Photos/p1.png)

Once the entire boundary is created press Enter to confirm the mask.

![Demo Image](./Photos/p2.png)

Once the mask looks good press Enter again.

The script will store the frames inside the `.Object_Reconstruction/input/{OBJECT_NAME}`
  ### Directory Structure

      {OBJECT_NAME}/
      ├── cam_K.txt
      ├── frame_count.txt
      ├── depth/
      ├── masks/
      └── rgb/
The name of the object will be updated inside `./Object_Reconstruction/last_object_name.txt`
> **Note:** The more frames you capture better results, try to capture about 600 frames for better results. 
## 4) Object Reconstruction

Run:
  ```bash
  ./run_conatiner.sh
  ```
Then inside the container(When running it for the first time or else you can skip this step):
  ```bash
  ./build.sh
  ```
Run the final script inside the docker:
  ```bash
  ./run.sh
  ```
  This will open the GUI where you can see the reconstruction of the object, once the reconstruction is complete the GUI will close itself.
  You can check the final mesh.obj file here: `.Object_Reconstruction/mesh/{OBJECT_NAME}/mesh/mesh_biggest_component_smoothed.obj`
  This OBJ file will be as input given to the Foundation Pose for 6D pose estimation.

  ![Demo Image](./Photos/p3.png)

> **Note:** Make sure to close the container moving onto 6D pose estimation.

## 5) Setting up Foundaion Pose

### Docker build
  ```bash
  cd live-pose/docker
  docker build --network host -t foundationpose .
  ```

### Install weights

using this link: https://drive.google.com/drive/folders/1wJayPZzZLZb6sxm6EeOQCJvzOAibJ693
Place them inside `.live-pose/FoundationPose/weights`

## 6) Implementing 6D pose estimation

### Running the container
```bash
cd docker
./run_doc_container.sh
```
### Building packages inside the container
```bash
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build.bash
  ```
### Running 6D pose estimation
Before running the script make sure that the object is kept stationery in front of the camera, and there should be no other object kept between  
```bash
./run_live.sh
```
Write the names of the objects you want to track e.g. duster, cup (if the object names are duster and cup) then press Enter.
GUI will pop-up for manual masking of the objects you choose.

![Demo Image](./Photos/p4.png) ![Demo Image](./Photos/p5.png)
> **Note:** Make sure that you mask only one object at a time in the window, terminal will print the object you have to mask in the current GUI.
> **Note:** Masking technique for Foundation-Pose is different than that of the Object Reconstruction.

![Demo Image](./Photos/p6.png)
![Demo Image](./Photos/p7.png)
  
## System Requirements for BundleSDF

To run **BundleSDF** effectively, especially for near real-time RGB-D object tracking and 3D reconstruction, your system should meet the following requirements:

### Recommended Setup

| Component | Specification |
|-----------|----------------|
| **GPU**   | NVIDIA GPU with CUDA support, **≥12 GB VRAM** (e.g., RTX 3080, A5000) |
| **CPU**   | 6–8 core processor (e.g., Intel i7/Ryzen 7, 3+ GHz) |
| **RAM**   | ≥32 GB system memory |
| **Storage** | SSD with sufficient space for video data and model checkpoints |
| **OS**    | Linux (Ubuntu 20.04+), Docker or Conda environment preferred |

### Minimum Requirements (May Be Unstable)

| Component | Minimum |
|-----------|---------|
| **GPU**   | NVIDIA GPU with CUDA & cuDNN, **≥8 GB VRAM** (e.g., RTX 2060/3060) |
| **CPU**   | Quad-core processor |
| **RAM**   | 16 GB |

### Notes

- CUDA and cuDNN must be properly installed and compatible with your PyTorch version.
- Low VRAM (e.g., 8 GB) may lead to out-of-memory errors or cuDNN failures.
- Start with small test sequences to verify setup before scaling to complex scenes.

## System Requirements for FoundationPose

To efficiently run the **FoundationPose** model—capable of 6D pose estimation and tracking for novel objects—ensure your system meets the following specs:

### Recommended Setup

| Component | Specification |
|-----------|----------------|
| **GPU**   | NVIDIA GPU with CUDA + cuDNN support (for accelerated inference) |
|           | - High-end desktop GPUs (e.g., RTX 3080 or better) for optimal throughput |
|           | - Jetson AGX Orin or similar for embedded use (running tracking at > 120 FPS) :contentReference[oaicite:1]{index=1} |
| **CPU**   | Multi-core processor (e.g., modern i7/i9 or Ryzen 7/9 or ARM equivalent in edge devices) |
| **RAM**   | ≥ 16 GB (more recommended for multiple concurrent models or complex scenes) |
| **Storage** | SSD with enough space for model files (~126 MB) and logs/data |
| **OS & Env** | Linux (Ubuntu 20.04+), CUDA & cuDNN matching your PyTorch or TensorRT version |

### Notes & Usage Tips

- The model is **compute-intensive**; typical GPU inference runs at lower frequency suitable for desktop applications :contentReference[oaicite:2]{index=2}.
- For real-time tracking workloads, Jetson Orin runs at **over 120 FPS** :contentReference[oaicite:3]{index=3}.
- Make sure your CUDA/cuDNN versions align with your framework (e.g., PyTorch or TensorRT).
- For CPU-only environments, expect much slower performance—tracking at high FPS requires GPU acceleration.

---

*Summary:*  
Use a CUDA-enabled NVIDIA GPU for efficient inference. For embedded/robotic scenarios, Jetson Orin-class devices are excellent. Ensure you have sufficient CPU, RAM, and correct CUDA/cuDNN versions.


