DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="MRNetData-${DATE}-ELNet+MRPyrNet"
DATA_PATH='/home/matteo/Downloads/MRNet-v1.0/'
NORM='contrast'
LR=1e-5
EPOCHS=200
SEED=$1 # --seed $SEED
SCHEDULER='plateau' #--lr_scheduler $SCHEDULER
PREFIX=ELNet+MRPyrNet

python3 train_mrpyrnet_mrnet.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type layer --lr=2e-5 --lr_scheduler None --seed $SEED

python3 train_mrpyrnet_mrnet.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type instance --lr=1.5e-5 --lr_scheduler None --seed $SEED # 1.5e-5
