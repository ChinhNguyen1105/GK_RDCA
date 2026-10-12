 python -m pip install --upgrade pip              
 python -m pip install numpy scipy scikit-learn matplotlib threadpoolctl pytest pandas 

# Iris
python evaluate_gk_rdca.py --dataset iris

# Wine
python evaluate_gk_rdca.py --dataset wine

# Breast Cancer
python evaluate_gk_rdca.py --dataset cancer

# Yeast
python evaluate_gk_rdca.py --dataset yeast

# Chạy tất cả
python evaluate_gk_rdca.py --dataset all