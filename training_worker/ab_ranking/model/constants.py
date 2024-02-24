# Pooling Strategy
AVERAGE_POOLING = 0
MAX_POOLING = 1
MAX_ABS_POOLING = 2

# Target Options
TARGET_1_AND_0 = 0
TARGET_1_ONLY = 1
TARGET_0_ONLY = 2

# Duplicate and Flip Options
DUPLICATE_AND_FLIP_ALL = 0
DUPLICATE_AND_FLIP_RANDOM = 1  # 50/50 chance to duplicate and flip

# Input types
EMBEDDING = "embedding"
EMBEDDING_POSITIVE = "embedding-positive"
EMBEDDING_NEGATIVE = "embedding-negative"
CLIP = "clip"
KANDINSKY_CLIP = "kandinsky-clip"
ALLOWED_INPUT_TYPES = [EMBEDDING, EMBEDDING_POSITIVE, EMBEDDING_NEGATIVE, CLIP, KANDINSKY_CLIP]