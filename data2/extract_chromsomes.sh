#!/usr/bin/env bash

folder_path="/data2/hic_lstm/data/GM12878_100kb"
hic_path="/data2/hic_lstm/data/GM12878_low/4DNFI9ZWZ5BS.hic"
slash="/"
extension=".txt"
filename="hic_chr"

chr_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)

for chr in "${chr_list[@]}"; do
  mkdir $folder_path$slash$chr
  txt_path="$folder_path$slash$chr$slash$filename$chr$extension"

  java -jar juicer_tools.jar dump observed KR $hic_path $chr $chr BP 100000 $txt_path
done
