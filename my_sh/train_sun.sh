export CUDA_VISIBLE_DEVICES=1
RATIO=0.01
LOG_DIR=results/train/sun_"${RATIO}"
LABELED_LIST=sunrgbd_v1_train_"${RATIO}"
DATASET=sunrgbd

mkdir -p "${LOG_DIR}"

for version in v4 v7 v9; do
    mkdir -p "${LOG_DIR}/${version}" && mkdir -p "${LOG_DIR}/${version}/eval"

    python -u train.py --log_dir="${LOG_DIR}/${version}" --data_ratio="${RATIO}" --dataset="${DATASET}" \
    --labeled_sample_list="${LABELED_LIST}_${version}.txt" --detector_checkpoint="results/pretrain/sun_${RATIO}/${version}/best_checkpoint_sum.tar" \
    --use_wandb \
    2>&1 | tee "${LOG_DIR}/${version}/LOG_ALL.log"

    python -u train.py --log_dir="${LOG_DIR}/${version}/eval" --data_ratio="${RATIO}" --dataset="${DATASET}" \
    --detector_checkpoint="${LOG_DIR}/${version}/best_checkpoint_sum.tar" \
    --labeled_sample_list="${LABELED_LIST}_${version}.txt" --use_iou_for_nms --eval
done
