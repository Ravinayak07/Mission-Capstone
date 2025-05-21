import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"             # Force CPU
os.environ["OMP_NUM_THREADS"] = "2"                   # Limit CPU threads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"              # Suppress logs
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"      # Prevent VRAM overload (if GPU is used)
