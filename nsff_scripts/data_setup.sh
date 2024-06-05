python preprocess_dynerf.py
python run_midas.py --data_path "../nerf_data/N3DV/set14/coffee_martini/dense/" --resize_height 1014
python run_flows_video.py --model models/raft-things.pth --data_path ../nerf_data/N3DV/set14/coffee_martini/dense/
python run_colmap_sparse_flow.py
python sparse_Flow_aggregator.py