DATE=$(date +"%Y-%m-%d-%H-%M")

DATA_PATH=[path-to-MRNet-v1.0]
EXPERIMENT="MRNetData-${DATE}-MRNet+MRPyrNet"
PREFIX=MRNet+MRPyrNet

python3 -W ignore train_mrpyrnet_mrnet.py -t acl -p sagittal --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX
python3 -W ignore train_mrpyrnet_mrnet.py -t acl -p coronal --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX
python3 -W ignore train_mrpyrnet_mrnet.py -t acl -p axial --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX

python3 -W ignore train_mrpyrnet_mrnet.py -t meniscus -p sagittal --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX
python3 -W ignore train_mrpyrnet_mrnet.py -t meniscus -p coronal --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX
python3 -W ignore train_mrpyrnet_mrnet.py -t meniscus -p axial --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX

python3 -W ignore train_logistic_regression.py --path_to_data $DATA_PATH --path_to_model "experiments/${EXPERIMENT}/models/"
