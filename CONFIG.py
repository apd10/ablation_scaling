"""Project configuration for architecture search / ablation experiments."""


# ##PHASE1: number of layers vs. parameters
KMEANS_NUM_FITS = 3

# hidden_size (y) vs layer_size_m (x)
KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE = 2

# num_experts (y) vs intermediate_size (x); both axes in raw units
NUM_EXPERTS_FIT_ANCHOR_INTERMEDIATE = 1024
NUM_EXPERTS_FIT_ANCHOR_NUM_EXPERTS = 1.0

# (layer_size_m, total_expert_hidden) coords excluded before k-means fits
FIT_OUTLIER_COORDS: tuple[tuple[float, int], ...] = (
    (1409.090909, 1_376_256),  # Nemotron Super 120B BF16/FP8
    (761.363636, 1_376_256),   # Nemotron Super 120B NVFP4
    (14131.147541, 1_179_648), # DeepSeek-V4-Pro
)
