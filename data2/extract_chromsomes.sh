#!/usr/bin/env bash

# download juicer tools first and specify right location of juicer_tools.jar in the last command 
# cd to desired folder 

wget -m https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined%2Ehic
mv ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_insitu_primary+replicate_combined.hic GM12878.hic
rm -r ftp.ncbi.nlm.nih.gov

hic_file_name="GM12878.hic"
extension=".txt"
output_filename="hic_chr"

chr_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)

for chr in "${chr_list[@]}"; do
  mkdir $chr
  txt_path="$chr"/"$output_filename$chr$extension"

  java -jar juicer_tools.jar dump observed KR $hic_file_name $chr $chr BP 100000 $txt_path
done
