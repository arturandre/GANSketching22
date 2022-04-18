import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm


#def create_lsun(save_dir, lmdb_dir, old_save_dir, resolution=256, max_images=None):
def create_lsun(save_dir, lmdb_dir, resolution=256, max_images=None):
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    import lmdb # pip install lmdb # pylint: disable=import-error
    import cv2 # pip install opencv-python
    import io
    #init_idx = len(os.listdir(old_save_dir)) # Files 
    init_idx = 0 # Files
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
            total_images = txn.stat()['entries']
            print("Total images: ", total_images)
            if max_images is None:
                max_images = total_images
            pbar = tqdm(total=max_images)
            pbar.update(init_idx)

            txn_cursor = txn.cursor()
            if not txn_cursor.set_range(str(init_idx).encode('utf8')):
                raise Exception(f"Could not start at idx: {init_idx}")

            

            #subpath = "820k_cat"
            #os.makedirs(os.path.join(save_dir, subpath), exist_ok=True)
            
            for _idx, (_key, value) in enumerate(txn_cursor, init_idx):
                pbar.update(1)
                if _idx == max_images:
                    break
                subidx = (_idx//10000)*10 # Changes every 10k indices
                subpath = f"{subidx}k_horse"
                os.makedirs(os.path.join(save_dir, subpath), exist_ok=True)
                #img_savename = os.path.join(save_dir, subpath,  '{:06d}.png'.format(_idx))
                img_savename = os.path.join(save_dir, subpath, '{:06d}.png'.format(_idx))
                if os.path.exists(img_savename):
                    continue
                try:
                    try:
                        #img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1) # deprecation warning
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                           img = np.asarray(Image.open(io.BytesIO(value)))
                    crop = np.min(img.shape[:2])
                    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                    img = Image.fromarray(img, 'RGB')
                    img = img.resize((resolution, resolution), Image.ANTIALIAS)
                    img.save(img_savename)
                except:
                    print(sys.exc_info()[1])
            pbar.close()


if __name__ == "__main__":
    save_dir = sys.argv[1]
    source = sys.argv[2]

    #old_save_dir = "./image/cat"
    #save_dir = "./image/cat_new"
    #source = "./image/lmdb/cat"
    os.makedirs(save_dir, exist_ok=True)
    #create_lsun(save_dir, source, old_save_dir)
    create_lsun(save_dir, source)
