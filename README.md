Benchmarks for AprilTag processing using a variety of implementations:
- libCuApriltags, the NVIDIA ISAAC ROS CUDA-accelerated AprilTag implementation. From GitHub NVIDIA-ISAAC-ROS/isaac\_ros\_nitros in isaac\_ros\_nitros/lib/cuapriltags/
- OpenCV ArUco, which is used by PhotonVision. Currently linked with "libopencv-dev" from apt, but it is probably worth checking out https://docs.opencv.org/4.x/d6/d15/tutorial\_building\_tegra\_cuda.html.
- AprilRobotics/apriltag, the reference CPU implementation

The samples in the samples/ directory are sourced from https://www.firstinspires.org/robotics/frc/playing-field under "Vision Sample Images."
