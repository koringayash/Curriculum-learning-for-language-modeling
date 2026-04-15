# 1. Setup
docker build -t nlp-curriculum .
docker run -it --gpus all -v $(pwd):/app nlp-curriculum bash

# 2. Dataset (Steps 1-5)
python Dataset/run.py 

# 3. Train Curriculum
python Model/run.py --mode curriculum

# 4. Train Random (Baseline)
python Model/run.py --mode random

# Resume if interrupted:
python Model/run.py --mode curriculum --resume

# 5. Compare
python Evaluation/run.py
