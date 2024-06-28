module load Anaconda/2021.05-nsc1
conda activate dgr-o3d-0.17

KITTI_CONFIG="OverlapPredator/configs/test/mbes/mbes_kitti.yaml"
KITTI_MODEL_PATH="weights/kitti.pth/"

INTERM_CONFIG="OverlapPredator/configs/test/mbes/mbes_trained_epoch10.yaml"
INTERM_MODEL_PATH="snapshot/20230912-sgd-lr.005/checkpoints/model_epoch_10.pth"

NOISE="crop"
OVERLAP="0.2"
MODEL="kitti"
COMPUTE=true
EVAL=true
TRANSFORM="pred"

if [[ $MODEL == "kitti" ]]; then
    NETWORK_CONFIG=$KITTI_CONFIG
    MODEL_PATH=$KITTI_MODEL_PATH
elif [[ $MODEL == "interm" ]]; then
    NETWORK_CONFIG=$INTERM_CONFIG
    MODEL_PATH=$INTERM_MODEL_PATH
fi

CONFIG_FOLDER="FCGF/mbes_data/configs/tests/meters"
MBES_CONFIG="$CONFIG_FOLDER/$NOISE/mbesdata_${NOISE}_meters_pairoverlap=$OVERLAP.yaml"
RESULTS_ROOT="snapshot/20230711-$NOISE-meters-pairoverlap=$OVERLAP/${MODEL_PATH}"
mkdir -p $RESULTS_ROOT

logname="$RESULTS_ROOT/mbes_test-$NOISE-$OVERLAP-$(basename $NETWORK_CONFIG .yaml).log"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="

if [[ $COMPUTE == true ]]; then
    echo "Running mbes_main.py on noise=$NOISE overlap=$OVERLAP, network=$NETWORK_CONFIG..."
    echo "Using mbes_config=$MBES_CONFIG..."
    echo "logging to $logname..."

    python OverlapPredator/mbes_main.py \
        --mbes_config  $MBES_CONFIG\
        --network_config $NETWORK_CONFIG \
        | tee $logname
fi

if [[ $EVAL == true ]]; then
    echo "======================================="
    echo "Evaluating results at $RESULTS_ROOT..."
    python mbes-registration-data/src/evaluate_results.py \
        --results_root $RESULTS_ROOT \
        --use_transforms $TRANSFORM \
        | tee $RESULTS_ROOT/eval-res-$NOISE-$OVERLAP-$TRANSFORM.log
fi
echo "Done!"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="
