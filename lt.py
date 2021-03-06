import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from Utils import load
from Models import apolo_resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import sampler
import utils
import numpy as np
import torchvision.models as models
import torch.nn.init as init
import time
import snip
import attack
import zenprune
import synflow

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    # choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'tiny-imagenet', 'cifar100'])
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--sv', dest='compute_sv', action='store_true',
                    help='compute_sv throughout training')
parser.add_argument('--ntk', dest='compute_ntk', action='store_true',
                    help='compute ntk eigenvalues throughout training')
parser.add_argument('--lrs',dest='compute_lrs',action='store_true',
                    help='compute number of linear regions throughout training')
parser.add_argument('--seed', default=1, type=int, help='seed')
# Following arguments are for pruning
parser.add_argument('--prune_method', type=str, default='NONE', choices=['NONE','RAND', 'SNIP', 'GRASP', 'Zen', 'Mag', 'Synflow'], help='Pruning methods for lottery ticket experiments.')
parser.add_argument('--prunesets_num', type=int, default=10, help='Number of datapoints for applying pruning methods.')
parser.add_argument('--sparse_iter', type=float, default=0, help='Sparsity level of neural networks.')
parser.add_argument('--sparse_lvl', type=float, default=1, help='Sparsity level of neural networks.')
parser.add_argument('--ONI', dest='ONI', action='store_true', help='set ONI on')
parser.add_argument('--T_iter', type=int, default=5, help='Number of iterations for ONI.')
parser.add_argument('--iter_prune', dest='iter_prune', action='store_true')
# Following arguments are for projection
parser.add_argument('--proj', dest='proj', action='store_true', help='set projection on')
parser.add_argument('--proj_freq', type=int, default=5, help='Apply projection every n iterations.')
parser.add_argument('--proj_clip_to', type=float, default=0.02, help='Smallest singular values clipped to.')
parser.add_argument('--ortho', dest='ortho', action='store_true', help='add orthogonal regularizer on.')

parser.add_argument('--pre_epochs', type=int, default=0, help='Number of pretraining epochs.')
parser.add_argument('--s_name', type=str, default='saved_sparsity', help='saved_sparsity.')
parser.add_argument('--s_value', type=float, default=1, help='given changing sparsity.')
parser.add_argument("--layer",  nargs="*",  type=int,  default=[],)
parser.add_argument('--structured', dest='structured', action='store_true', help='set structured masks')
parser.add_argument('--reduce_ratio', type=float, default=1, help='compact masks into reduce_ratio x 100% number of channels.')
parser.add_argument('--shuffle_ratio', type=float, default=0.1, help='shuffle ratio of structured pruning.')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale weight after pruning')
parser.add_argument('--adv', dest='adv', action='store_true', help='If using adversarial trick')
parser.add_argument('--ep_coe', type=float, default=0, help='coefficient for expressivity.')
parser.add_argument('--ortho_importance', type=float, default=0, help='orthogonality regularizer importance.')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.sparse_lvl = 0.8 ** args.sparse_iter
    print(args.sparse_lvl)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    if args.dataset =='cifar10':
        print('Loading {} dataset.'.format(args.dataset))
        input_shape, num_classes = load.dimension(args.dataset) 
        train_dataset, train_loader = load.dataloader(args.dataset, args.batch_size, True, args.workers)
        _, val_loader = load.dataloader(args.dataset, 128, False, args.workers)

    elif args.dataset == 'tiny-imagenet':
        args.batch_size = 256
        args.lr = 0.2
        args.epochs = 200
        print('Loading {} dataset.'.format(args.dataset))
        input_shape, num_classes = load.dimension(args.dataset) 
        train_dataset, train_loader = load.dataloader(args.dataset, args.batch_size, True, args.workers)
        _, val_loader = load.dataloader(args.dataset, 128, False, args.workers)

    elif args.dataset == 'cifar100':
        args.batch_size = 128
        # args.lr = 0.01
        args.epochs = 160
        # args.weight_decay = 5e-4
        input_shape, num_classes = load.dimension(args.dataset) 
        train_dataset, train_loader = load.dataloader(args.dataset, args.batch_size, True, args.workers)
        _, val_loader = load.dataloader(args.dataset, 128, False, args.workers)

    if args.arch == 'resnet20':
        print('Creating {} model.'.format(args.arch))
        # model = torch.nn.DataParallel(resnet.__dict__[args.arch](ONI=args.ONI, T_iter=args.T_iter))
        model = resnet.__dict__[args.arch](ONI=args.ONI, T_iter=args.T_iter)
        model.cuda()
    elif args.arch == 'resnet18':
        print('Creating {} model.'.format(args.arch))
        # Using resnet18 from Synflow
        # model = load.model(args.arch, 'tinyimagenet')(input_shape, 
        #                                              num_classes,
        #                                              dense_classifier = True).cuda()
        # Using resnet18 from torchvision
        model = models.resnet18()
        model.fc = nn.Linear(512, num_classes)
        model.cuda()
        utils.kaiming_initialize(model)
    elif args.arch == 'resnet110' or args.arch == 'resnet110full':
        # Using resnet110 from Apollo
        # model = apolo_resnet.ResNet(110, num_classes=num_classes)
        model = load.model(args.arch, 'lottery')(input_shape, 
                                             num_classes,
                                             dense_classifier = True).cuda()
    elif args.arch in ['vgg16full', 'vgg16full-bn', 'vgg11full', 'vgg11full-bn'] :
        if args.dataset == 'tiny-imagenet':
            modeltype = 'tinyimagenet'
        else:
            modeltype = 'lottery'
        # Using resnet110 from Apollo
        # model = apolo_resnet.ResNet(110, num_classes=num_classes)
        model = load.model(args.arch, modeltype)(input_shape, 
                                             num_classes,
                                             dense_classifier = True).cuda()
    
    # for layer in model.modules():
    #     if isinstance(layer, nn.Linear):
    #         init.orthogonal_(layer.weight.data)
    #     elif isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
    #         special_init.DeltaOrthogonal_init(layer.weight.data)

    print('Number of parameters of model: {}.'.format(count_parameters(model)))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.compute_sv:
        print('[*] Will compute singular values throught training.')
        size_hook = utils.get_hook(model, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))
        utils.run_once(train_loader, model)
        utils.detach_hook([size_hook])
        training_sv = []
        training_svmax = []
        training_sv20 = [] # 50% singular value
        training_sv50 = [] # 50% singular value
        training_sv80 = [] # 80% singular value
        training_kclip = [] # singular values larger than 1e-12
        sv, svmax, sv20, sv50, sv80, kclip = utils.get_sv(model, size_hook)
        training_sv.append(sv)
        training_svmax.append(svmax)
        training_sv20.append(sv20)
        training_sv50.append(sv50)
        training_sv80.append(sv80)
        training_kclip.append(kclip)
    
    if args.compute_ntk:
        training_ntk_eig = []
        if num_classes>=32:
            _, ntk_loader = load.dataloader(args.dataset, 32, True, args.workers)
            grasp_fetch = False
        else:
            ntk_loader = train_loader
            grasp_fetch = True
        training_ntk_eig.append(utils.get_ntk_eig(ntk_loader, [model], train_mode = True, num_batch=1, num_classes=num_classes, samples_per_class=1, grasp_fetch=grasp_fetch))
    
    if args.compute_lrs:
        # training_lrs = []
        # lrc_model = utils.Linear_Region_Collector(train_loader, input_size=(args.batch_size,*input_shape), sample_batch=300)
        # lrc_model.reinit(models=[model])
        # lrs = lrc_model.forward_batch_sample()[0]
        # training_lrs.append(lrs)
        # lrc_model.clear_hooks()
        # print('[*] Current number of linear regions:{}'.format(lrs))
        GAP_zen, output_zen = utils.get_zenscore(model, train_loader, args.arch, num_classes)
        print('[*] Before pruning: GAP_zen:{:e}, output_zen:{:e}'.format(GAP_zen,output_zen))
    
    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov = True,
                                weight_decay=args.weight_decay)

    if args.dataset ==  'tiny-imagenet':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
                                                        # milestones=[30, 60, 80], last_epoch=args.start_epoch - 1)
    elif args.dataset ==  'cifar100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60, 120], gamma = 0.2, last_epoch=args.start_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[80, 120], last_epoch=args.start_epoch - 1)

    # This part is for training full NN model to obtain Lottery ticket

    # # First save original network:
    init_path = os.path.join(args.save_dir, 'init_checkpoint.th')
    save_checkpoint({
                'state_dict': model.state_dict()
            }, False, filename=init_path)

    if args.prune_method == 'NONE':
        pre_epochs = args.epochs
    else:
        pre_epochs = 0

    training_loss = []
    for epoch in range(pre_epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, track = training_loss)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'densenet_checkpoint.th'))

        if args.compute_sv and epoch % args.save_every == 0:
            sv,  svmax, sv20, sv50, sv80, kclip= utils.get_sv(model, size_hook)
            training_sv.append(sv)
            training_svmax.append(svmax)
            training_sv20.append(sv20)
            training_sv50.append(sv50)
            training_sv80.append(sv80)
            training_kclip.append(kclip)
            np.save(os.path.join(args.save_dir, 'sv.npy'), training_sv)
            np.save(os.path.join(args.save_dir, 'sv_svmax.npy'), training_svmax)
            np.save(os.path.join(args.save_dir, 'sv_sv20.npy'), training_sv20)
            np.save(os.path.join(args.save_dir, 'sv_sv50.npy'), training_sv50)
            np.save(os.path.join(args.save_dir, 'sv_sv80.npy'), training_sv80)
            np.save(os.path.join(args.save_dir, 'sv_kclip.npy'), training_kclip)

        if args.compute_ntk and epoch % args.save_every == 0:
            training_ntk_eig.append(utils.get_ntk_eig(ntk_loader, [model], train_mode = True, num_batch=1, num_classes=num_classes, samples_per_class=1, grasp_fetch=grasp_fetch))
            np.save(os.path.join(args.save_dir, 'ntk_eig.npy'), training_ntk_eig)
        


    print('[*] {} epochs of dense network pre-training done'.format(pre_epochs))
    np.save(os.path.join(args.save_dir, 'trainloss.npy'), training_loss)

    # densenet_checkpoint = torch.load(os.path.join(args.save_dir, 'densenet_checkpoint.th'))
    # model.load_state_dict(densenet_checkpoint['state_dict'])
    # print('Model loaded!')
    # Obtain lottery ticket by magnitude pruning
    if args.prune_method == 'NONE':
        snip.apply_mag_prune(args, model)
        # reinitialize
        init_checkpoint = torch.load(init_path)
        model.load_state_dict(init_checkpoint['state_dict'])
        print('Model reinitialized!')
    elif args.prune_method == 'SNIP':
        init_checkpoint = torch.load(init_path)
        model.load_state_dict(init_checkpoint['state_dict'])
        print('Model reinitialized!')
        snip.apply_snip(args, [model], train_loader, criterion, num_classes=num_classes)
        # attack.shuffle_mask(model)
    elif args.prune_method == 'RAND':
        init_checkpoint = torch.load(init_path)
        model.load_state_dict(init_checkpoint['state_dict'])
        print('Model reinitialized!')
        snip.apply_rand_prune([model], args.sparse_lvl)
    elif args.prune_method == 'GRASP':
        init_checkpoint = torch.load(init_path)
        model.load_state_dict(init_checkpoint['state_dict'])
        print('Model reinitialized!')
        snip.apply_grasp(args, [model], train_loader, criterion, num_classes=num_classes)
    elif args.prune_method == 'Zen':
        zenprune.apply_zenprune(args, [model], train_loader)
        # zenprune.apply_cont_zenprune(args, [model], train_loader)
        # zenprune.apply_zentransfer(args, [model], train_loader)
        # init_checkpoint = torch.load(init_path)
        # model.load_state_dict(init_checkpoint['state_dict'])
        # print('Model reinitialized!')
    elif args.prune_method == 'Mag':
        snip.apply_mag_prune(args, model)
        init_checkpoint = torch.load(init_path)
        model.load_state_dict(init_checkpoint['state_dict'])
        print('Model reinitialized!')
    elif args.prune_method == 'Synflow':
        synflow.apply_synflow(args, model)

    print('{} done, sparsity of the current model: {}.'.format(args.prune_method, utils.check_sparsity(model)))
    
    if args.compute_lrs:
        # training_lrs = []
        # lrc_model = utils.Linear_Region_Collector(train_loader, input_size=(args.batch_size,*input_shape), sample_batch=300)
        # lrc_model.reinit(models=[model])
        # lrs = lrc_model.forward_batch_sample()[0]
        # training_lrs.append(lrs)
        # lrc_model.clear_hooks()
        # print('[*] Current number of linear regions:{}'.format(lrs))
        GAP_zen, output_zen = utils.get_zenscore(model, train_loader, args.arch, num_classes)
        print('[*] After pruning: GAP_zen:{:e}, output_zen:{:e}'.format(GAP_zen,output_zen))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # Recreate optimizer and learning scheduler
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov = True,
                                weight_decay=args.weight_decay)

    if args.dataset ==  'tiny-imagenet':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
                                                        # milestones=[30, 60, 80], last_epoch=args.start_epoch - 1)
    elif args.dataset ==  'cifar100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60, 120], gamma = 0.2, last_epoch=args.start_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[80, 120], last_epoch=args.start_epoch - 1)

    for epoch in range(args.epochs):
    # for epoch in range(args.pre_epochs, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, track=training_loss, ortho=args.ortho)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'sparsenet_checkpoint.th'))
            # if args.prune_method !='NONE':
            #     print(utils.check_layer_sparsity(model))   

        # save_checkpoint({
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        # }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

        if args.compute_sv and epoch % args.save_every == 0:
            sv,  svmax, sv20, sv50, sv80,  kclip = utils.get_sv(model, size_hook)
            training_sv.append(sv)
            training_svmax.append(svmax)
            training_sv20.append(sv20)
            training_sv50.append(sv50)
            training_sv80.append(sv80)
            training_kclip.append(kclip)
            np.save(os.path.join(args.save_dir, 'sv.npy'), training_sv)
            np.save(os.path.join(args.save_dir, 'sv_svmax.npy'), training_svmax)
            np.save(os.path.join(args.save_dir, 'sv_sv20.npy'), training_sv20)
            np.save(os.path.join(args.save_dir, 'sv_sv50.npy'), training_sv50)
            np.save(os.path.join(args.save_dir, 'sv_sv80.npy'), training_sv80)
            np.save(os.path.join(args.save_dir, 'sv_kclip.npy'), training_kclip)

        if args.compute_ntk and epoch % args.save_every == 0:
            training_ntk_eig.append(utils.get_ntk_eig(ntk_loader, [model], train_mode = True, num_batch=1, num_classes=num_classes, samples_per_class=1, grasp_fetch=grasp_fetch))
            np.save(os.path.join(args.save_dir, 'ntk_eig.npy'), training_ntk_eig)

        # if args.compute_lrs and epoch % args.save_every == 0:
        #     lrc_model.reinit(models=[model])
        #     lrs = lrc_model.forward_batch_sample()[0]
        #     training_lrs.append(lrs)
        #     lrc_model.clear_hooks()
        #     print('[*] Current number of linear regions:{}'.format(lrs))
    np.save(os.path.join(args.save_dir, 'trainloss.npy'), training_loss)

def train(train_loader, model, criterion, optimizer, epoch, track=None, ortho=False):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # add orthogonal loss
        if ortho:
            # start_time = time.time()
            loss += 0.0001*svfp.get_svip_loss_with_target(model, args.layer)
            # print(time.time()-start_time)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    print('training accuracy: {:.3f}'.format(top1.avg))
    
    if track is not None:
        track.append(losses.avg)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    main()
