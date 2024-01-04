# STEP 0: prepare csv
# You need to put all candidate samples into a csv, containing the `job_uuid` or `file_path` column. This list should already be filtered, as there will be no further filtering.

# please use local IP when run it on devbox

# STEP 1: get job info

# if you have file_path, but no job_uuid, run this code
# python download_job_info_by_uuid.py --csv-path=./image_list.csv --api-addr=123.176.98.90:8764 

# if you alread have uuid, but no file_path, run this code
# python download_uuid_by_file_path.py --csv-path=./image_list.csv --minio-addr=123.176.98.90:9000 --minio-access-key=GXvqLWtthELCaROPITOG --minio-secret-key=DmlKgey5u0DnMHP30Vg7rkLT0NNbNIGaM8IwPckD

# otherwise, go to step2

# STEP 2: get ranking model
python download_latest_ranking_model.py --dataset=environmental --input-type=clip --output=./ --api-addr=123.176.98.90:8764 --minio-addr=123.176.98.90:9000 --minio-access-key=GXvqLWtthELCaROPITOG --minio-secret-key=DmlKgey5u0DnMHP30Vg7rkLT0NNbNIGaM8IwPckD

# STEP 3: assign sigma_score and cluster_id
python calculate_score_and_cluster_by_file_path.py --csv-path=./image_list.csv --scoring-model-path=./2023-12-22-00-score-linear-clip.safetensors --pca-model-path=../data/pca.npz --kmeans-model-path=../data/kmeans.npz --minio-addr=123.176.98.90:9000 --minio-access-key=GXvqLWtthELCaROPITOG --minio-secret-key=DmlKgey5u0DnMHP30Vg7rkLT0NNbNIGaM8IwPckD

# STEP 4: make pairs within same sigma score bin
python get_pairs_within_same_sigma_score_bin.py --csv-path=./image_list.csv --output-path=./same_sigma_score_bin_pairs.json --bins=10 --pairs=100 --bin-type=quantile

# STEP 5: make pairs within same clutser
python get_pairs_within_same_cluster.py --csv-path=./image_list.csv --output-path=./same_cluster_pairs.json --pairs=100 --cluster-type=cluster_id_48

# STEP 6: marge pairs
python merge_pairs.py ./same_sigma_score_bin_pairs.json ./same_cluster_pairs.json --output-path=./pairs.json
