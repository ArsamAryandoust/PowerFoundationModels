#!/bin/bash
: '
Compress processed OPFData files for upload to Hugging Face.

Example run:

    $ compress_opfdata.sh regular pglib_opf_case14_ieee

    $ compress_opfdata.sh n-2 pglib_opf_case118_ieee

List of cases:

    pglib_opf_case14_ieee
    pglib_opf_case30_ieee
    pglib_opf_case57_ieee
    pglib_opf_case118_ieee
    pglib_opf_case500_goc
    pglib_opf_case2000_goc
    pglib_opf_case4661_sdet
    pglib_opf_case6470_rte
    pglib_opf_case10000_goc
    pglib_opf_case13659_pegase

'
# base path to processed OPFData repository
PATH_BASE="../../donti_group_shared/AI4Climate/processed/OPFData"

echo "realease": $1
echo "grid": $2

# create path to release
if [ $1 == "regular" ]; then
    PATH_DATA="${PATH_BASE}/dataset_release_1"
elif [ $1 == "n-1" ]; then
    PATH_DATA="${PATH_BASE}/dataset_release_1_nminusone"
else
    echo "invalid first argument"
fi

# create path to dataset
PATH_DATA="${PATH_DATA}/$2"

for entry in $(ls "$PATH_DATA")
do
    PATH_DATA_GROUP="${PATH_DATA}/${entry}"
    PATH_FILENAME="${PATH_DATA_GROUP}.tar.gz"
    tar -czvf "$PATH_FILENAME" "$PATH_DATA_GROUP" &
done 

# Wait until all parallel background processes complete
wait

echo "Compression script ran successfully!"