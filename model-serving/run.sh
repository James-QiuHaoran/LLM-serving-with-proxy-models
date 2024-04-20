# no batching
export MODE_ENV="NO_BATCHING"
export BATCH_SIZE_ENV="1"
export PER_ROUND_ENV="true"
rounds=(0 1 2 3 4)

for round in "${rounds[@]}"; do
    export TURN_ID_ENV="$round"
    python auto_eval.py
    python auto_eval_lineplot.py
    done

# static batching
export MODE_ENV="STATIC_BATCHING"
export PER_ROUND_ENV="true"
batch_sizes=(2 4 6 8 10)
rounds=(0 1 2 3 4)

for batch_size in "${batch_sizes[@]}"; do
    for round in "${rounds[@]}"; do
        export BATCH_SIZE_ENV="$batch_size"
        export TURN_ID_ENV="$round"
        python auto_eval.py
        python auto_eval_lineplot.py
    done
done

# dynamic batching
export MODE_ENV="DYNAMIC_BATCHING"
export DYNAMIC_BATCHING_ENV="true"
export PER_ROUND_ENV="true"
batch_sizes=(2 4 6 8 10)
rounds=(0 1 2 3 4)

for batch_size in "${batch_sizes[@]}"; do
    for round in "${rounds[@]}"; do
        export BATCH_SIZE_ENV="$batch_size"
        export TURN_ID_ENV="$round"
        python auto_eval.py
        python auto_eval_lineplot.py
    done
done
