#~/bin/bash

module load java

for i in {1..22}
do
  chr="chr$i"
  size="$(awk -v chr="$chr" '$1==chr {print $2}' hg19.chrom.sizes)"
  out="GM12878/chr$i.txt"
  echo $out
  java -jar juicer_tools.jar dump observed KR https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic $i:0:$size $i:0:$size BP 10000 $out
done 