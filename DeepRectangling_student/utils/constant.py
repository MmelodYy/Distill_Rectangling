# define the mesh resolution.
# By the way, for a fair comparison with DeepRectangling, we used the same mesh size.
# If you want to improve model performance quickly, directly increasing the number of meshes will result in a significant performance improvement.
# In addition, the improvement of FID metrics can be realized by applying SSIM loss.
GRID_W = 8
GRID_H = 6

# define the GPU device
GPU_DEVICE = 0

