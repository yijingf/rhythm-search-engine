#!/bin/bash
root_dir="/isi/music/yijing/maestro-v3.0.0/audio"

# output_dir="./similar_segment/cos"
# output_dir="./similar_segment/dpw"
output_dir="./dpw"
rm -rf $output_dir
mkdir -p $output_dir

# ranking="cos_rank.csv"
ranking="dpw_rank.csv"

i=0

# sed 1d $ranking | while IFS="," read -r fname t_start; do
sed 1d $ranking | while IFS=$'\t' read -r composer title fname t_start; do 
    echo $title by $composer

    input_fname=$root_dir/$fname
    output_fname=$output_dir/$(printf "%02d" $i).wav


    ffmpeg -loglevel error -ss $t_start -t 10 -i $input_fname $output_fname

    echo "---------"

    i=$((i+1))
    # echo $fname
    # echo $t_start

done

