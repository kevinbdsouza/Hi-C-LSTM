import numpy as np
import pandas as pd 

sizes = pd.read_csv("/data2/hic_lstm/data/hg19.chrom.sizes", header=None, names=['chr', 'len'], sep='\t')
sizes_new = {}

total = 0
for i in range(1, 23):
    chr = 'chr%s' % str(i)
    length = sizes.loc[sizes['chr']==chr]['len']
    length = int(np.floor(length/10000))

    total+=length
    sizes_new[chr] = total

np.save("/data2/hic_lstm/data/chr_cum_sizes2.npy", sizes_new)
print(total)