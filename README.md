# Real-time-6D-Pose-Estimation-with-On-the-Fly-Object-Reconstruction
## 0) Clone the repository in your system

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

## 2) Docker Setup

To set up the Docker environment, run the following command:

```bash
docker build --network host -t nvcr.io/nvidian/bundlesdf .
```
## 3) Capturing the data (To be done outside docker)
- The data is captured on Intel RealSense D435I.

  ### why this camera ?
  - it provides synchronized depth, RGB, and IMU (accelerometer + gyroscope) data, enabling accurate 3D reconstruction and camera pose estimation. Its wide field of view and high-resolution depth sensing make it ideal for dense mapping and visual-inertial odometry tasks.

- To capture the data
   ```bash
   cd Object_Reconstruction
   ./capture.sh
   ```
   - Enter the name of the object you want to reconstruct.
   - this will open a masking window where you have to manually mask the image shown on the gui by clicking on the points around the boundary of the object as shown in the image.
   - ![Demo Image](./assets/demo.png)
   - Once the entire boundary is created press Enter to confirm the mask.
   - ![Demo Image](./assets/demo.png)
   - Once the mask looks good press Enter again.
   - The script will store the frames inside the `.Object_Reconstruction/input/{OBJECT_NAME}`
  ### Directory Structure

      {OBJECT_NAME}/
      ├── cam_K.txt
      ├── frame_count.txt
      ├── depth/
      ├── masks/
      └── rgb/
  - The name of the object will be updated inside `./Object_Reconstruction/last_object_name.txt`
 
## 4) Object Reconstruction

- Run:
  ```bash
  ./run_conatiner.sh
  ```
- Then inside the container(When running it for the first time or else you can skip this step):
  ```bash
  ./build.sh
  ```
- Run the final script inside the docker:
  ```bash
  ./run.sh
  ```
  This will open the GUI where you can see the reconstruction of the object, once the reconstruction is complete the GUI will close itself.
  You can check the final mesh.obj file here: `.Object_Reconstruction/mesh/{OBJECT_NAME}/mesh/mesh_biggest_component_smoothed.obj`
  This OBJ file will be as input given to the Foundation Pose for 6D pose estimation.

  ![Demo Image](./assets/demo.png)

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
> **Note:** Make sure that you mask only one object at a time in the window, terminal will print the object you have to mask in the current GUI.
> **Note:** Masking technique for Foundation-Pose is different than that of the Object Reconstruction.


  

  
