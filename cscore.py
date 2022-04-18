import sys
from cleanfid import fid

def logfile_pngs(logfile, outfolder):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    import pandas as pd
    import io
    import os
    import re

    header = "epoch,iters,optim time,data time,G_sketch,G_image,D_fake_sketch,D_real_sketch,D_fake_image,D_real_image,D_r1_loss_sketch,D_r1_loss_image"

    with open(logfile, "r") as lf:
        lf.readline()
        s = lf.read()
        s = s.replace('(', '').replace(')', '')
        for h in header.split(','):
            s = s.replace(f"{h}:", '')
        s = s.replace(',', ' ')
        s = re.sub('[ ]+', ' ', s)
        s = re.sub('[ ]*\n[ ]*', '\n', s)
        s = s.strip()
        s = s.replace(' ', ',')
        
        s = pd.read_csv(io.StringIO(s), names=header.split(','))
            

    for c in s.columns:
        if c == "G_image":
            s[c][s[c] > 20] = -1
        plt.scatter(s['iters'], s[c])
        plt.plot(s['iters'], s[c])
        plt.title('sup 30 10k s10')
        plt.ylabel(c)
        plt.xlabel('iterations')
        #plt.savefig(f'/scratch/arturao/GANSketching22/checkpoint/horse_riders_original_sup30-10000-s10-ft/{c.strip()}.png')
        plt.savefig(os.path.join(outfolder, f'{c}.png'))
        plt.clf()

def computeFID(rp_exp, fp_exp):
    #rp = f"/scratch/arturao/GANSketching_old/data/eval/gabled_church/image"
    rp = f"/scratch/arturao/GANSketching_old/data/eval/{rp_exp}/image"
    #rp = f"/scratch/arturao/GANSketching_old/data/eval/horse_riders/image"
    #fp = f"/scratch/arturao/GANSketching22/output/horse_riders_original30-800-7500/"
    fp = f"/scratch/arturao/GANSketching22/output/{fp_exp}/"
    score = fid.compute_fid(rp,fp, mode="clean")
    print(f"FID score: {score}")

if __name__ == "__main__":
    computeFID(sys.argv[1], sys.argv[2])
    #logfile_pngs(sys.argv[1], sys.argv[2])
     
