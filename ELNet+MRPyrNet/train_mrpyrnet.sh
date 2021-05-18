DATE=$(date +"%Y-%m-%d-%H-%M")

DATA_PATH=[path-to-MRNet-v1.0]
EXPERIMENT="MRNetData-${DATE}-ELNet+MRPyrNet"
PREFIX=ELNet+MRPyrNet

python3 -W ignore train_mrpyrnet_mrnet.py -t acl -p axial --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX --set_norm_type layer --lr=2e-5 --D 6
python3 -W ignore train_mrpyrnet_mrnet.py -t meniscus -p coronal --experiment $EXPERIMENT --path_to_data $DATA_PATH --prefix_name $PREFIX --set_norm_type instance --lr=1.5e-5 --D 7
