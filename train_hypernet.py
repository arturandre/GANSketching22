
print("import time")
import itertools
print("itertools")
import time
print("from typing import Dict, List, Optional, Tuple, Union, Any  # noqa")
from typing import Dict, List, Optional, Tuple, Union, Any  # noqa
print("import torch")
import torch
print("import torch.nn as nn")
import torch.nn as nn
print("import torch.multiprocessing as mp")
import torch.multiprocessing as mp
print("from hyper_net import HyperNet")
from hyper_net import HyperNet

print("from copy import deepcopy")
from copy import deepcopy
print("from training.gan_model import get_param_by_name")
from training.gan_model import get_param_by_name


print("from options import get_opt, print_options")
from options import get_opt, print_options
print("from eval import Evaluator")
from eval import Evaluator
print("from util.visualizer import Visualizer")
from util.visualizer import Visualizer
print("from training.gan_trainer import GANTrainer")
from training.gan_trainer import GANTrainer
print("from training.dataset import create_dataloader, yield_data")
from training.dataset import create_dataloader, yield_data

print("from tqdm import tqdm")
from tqdm import tqdm

import os

herecounter = 0
def printhere():
    global herecounter
    herecounter += 1
    print(f"Here {herecounter}")


def training_loop():
    torch.backends.cudnn.benchmark = True
    printhere()

    opt, parser = get_opt()
    printhere()
    opt.isTrain = True
    printhere()

    # needs to switch to spawn mode to be compatible with evaluation
    if not opt.disable_eval:
        mp.set_start_method('spawn')
    printhere()

    
    if opt.dataroot_sketch_augs is not None:
        #sketch_augs_set = "/scratch/arturao/GANSketching22/data/sketch/photosketch/horse_riders_augs"
        def meta_create_dataloader(dataroot_sketch_augs, index):
            all_sketch_folders = os.listdir(dataroot_sketch_augs)
            circular_index = index % len(all_sketch_folders)
            dataroot_sketch = os.path.join(dataroot_sketch_augs, all_sketch_folders[circular_index])
            dataloader_sketch, sampler_sketch = create_dataloader(dataroot_sketch,
                                                                opt.size,
                                                                opt.batch,
                                                                opt.sketch_channel)
            return dataloader_sketch, sampler_sketch
    else:
        # dataloader for user sketches
        dataloader_sketch, sampler_sketch = create_dataloader(opt.dataroot_sketch,
                                                            opt.size,
                                                            opt.batch,
                                                            opt.sketch_channel)
    printhere() # Here 9
    # Ref: /scratch/arturao/3FGAN/cfgs/ImageNet/meta_train/SNGAN128_4bn1ccbn_dog_t32s5.yaml
    # dataloader for image regularization
    if opt.dataroot_image is not None:
        print(opt.dataroot_image)
        dataloader_image, sampler_image = create_dataloader(opt.dataroot_image,
                                                            opt.size,
                                                            opt.batch)
        data_yield_image = yield_data(dataloader_image, sampler_image)
    
    # support_imgs = dataloader_sketch
    # pred_for_support = []
    # printhere()
    # for i, data_sketch in enumerate(support_imgs):
    #     print(i)
    #     data_sketch = data_sketch.cuda()
    #     preds = hyper_net(data_sketch)
    #     pred_for_support.append(preds)
    #     pred_weights, pred_biases = preds[0], preds[1]
    #     print(len(pred_weights), len(pred_biases))

    trainer = GANTrainer(opt)
    printhere()
        
    print_options(parser, opt)

    trainer.gan_model.print_trainable_params()
    if not opt.disable_eval:
        evaluator = Evaluator(opt, trainer.get_gan_model())
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    # the total number of training iterations
    if opt.resume_iter is None:
        total_iters = 0
    else:
        total_iters = opt.resume_iter

    optimize_time = 0.1
    print("The training ends when either max_epochs or max_iters is reached.")
    iterbar = tqdm(total=opt.max_iter, position=total_iters, leave=False, desc="total_iters")
    for epoch in tqdm(range(opt.max_epoch), position=0, desc="epoch"):
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        if opt.dataroot_sketch_augs is not None:
            #sketch_augs_set = "/scratch/arturao/GANSketching22/data/sketch/photosketch/horse_riders_augs"
            dataloader_sketch, sampler_sketch = meta_create_dataloader(opt.dataroot_sketch_augs, epoch)

        for i, data_sketch in enumerate(dataloader_sketch):  # inner loop within one epoch
            if total_iters >= opt.max_iter:
                iterbar.close()
                return

            # makes dictionary to store all inputs
            data = {}
            data['sketch'] = data_sketch
            if opt.dataroot_image is not None:
                data_image = next(data_yield_image)
                data['image'] = data_image

            # timer for data loading per iteration
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # timer for optimization per iteration
            optimize_start_time = time.time()
            trainer.train_one_step(data, total_iters)
            optimize_time = (time.time() - optimize_start_time) * 0.005 + 0.995 * optimize_time

            # print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, total_iters, losses, optimize_time, t_data)
                visualizer.plot_current_errors(losses, total_iters)

            # display images on wandb and save images to a HTML file
            if total_iters % opt.display_freq == 0:
                visuals = trainer.get_visuals()
                visualizer.display_current_results(visuals, epoch, total_iters)

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                trainer.save(total_iters)

            # evaluate the latest model
            if not opt.disable_eval and total_iters % opt.eval_freq == 0:
                metrics_start_time = time.time()
                metrics, best_so_far = evaluator.run_metrics(total_iters)
                metrics_time = time.time() - metrics_start_time

                visualizer.print_current_metrics(epoch, total_iters, metrics, metrics_time)
                visualizer.plot_current_errors(metrics, total_iters)

            total_iters += 1
            iterbar.update(1)
            epoch_iter += 1
            iter_data_time = time.time()
    iterbar.close()

if __name__ == "__main__":
    training_loop()
    print('Training was successfully finished.')
