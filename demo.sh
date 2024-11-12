#!/bin/sh

### common settings for synthetic examples ###
categorical_idxs="Entity1/Entity2/Entity3"
time_idx="Time" 
freq="H"
k=10
width=10
init_len=20
##############################################

# Synthetic 121
python main.py \
    --input_fpath "./_dat/sample_121.csv.gz" \
    --out_dir "./_out/sample_121/" \
    --pattern "121" \
    --categorical_idxs $categorical_idxs \
    --time_idx $time_idx \
    --freq $freq \
    --k $k \
    --width $width \
    --init_len $init_len \
    --anomaly
    # --verbose

# Synthetic 12321
python main.py \
    --input_fpath "./_dat/sample_12321.csv.gz" \
    --out_dir "./_out/sample_12321/" \
    --pattern "12321" \
    --categorical_idxs $categorical_idxs \
    --time_idx $time_idx \
    --freq $freq \
    --k $k \
    --width $width \
    --init_len $init_len \
    --anomaly
    # --verbose

# Synthetic 12341234
python main.py \
    --input_fpath "./_dat/sample_12341234.csv.gz" \
    --out_dir "./_out/sample_12341234/" \
    --pattern "12341234" \
    --categorical_idxs $categorical_idxs \
    --time_idx $time_idx \
    --freq $freq \
    --k $k \
    --width $width \
    --init_len $init_len \
    --anomaly
    # --verbose

# Synthetic 12213331
python main.py \
    --input_fpath "./_dat/sample_12213331.csv.gz" \
    --out_dir "./_out/sample_12213331/" \
    --pattern "12213331" \
    --categorical_idxs $categorical_idxs \
    --time_idx $time_idx \
    --freq $freq \
    --k $k \
    --width $width \
    --init_len $init_len \
    --anomaly
    # --verbose