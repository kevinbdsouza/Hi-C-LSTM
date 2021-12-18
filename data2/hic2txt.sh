#~/bin/bash

module load java

for i in {1..22}
do
  mkdir "$i"
  out="$i/hic_chr$i.txt"
  echo $out
  java -jar juicer_tools.jar dump observed KR ENCFF355OWW.hic $i $i BP 10000 $out
done
