import snip
import resnet
import torch.nn.parallel
import torch.nn as nn
import utils

def apply_snipres(args, nets, data_loader, criterion, input_shape, num_classes, samples_per_class = 10):
    if args.arch == 'resnet20':
        print('Creating {} model.'.format(args.arch))
        cutmodel = torch.nn.DataParallel(resnet.__dict__[args.arch](ONI=args.ONI, cut=True, T_iter=args.T_iter))
        cutmodel.cuda()
    # elif args.arch == 'resnet18':
    #     print('Creating {} model.'.format(args.arch))
    #     # Using resnet18 from Synflow
    #     cutmodel = load.model(args.arch, 'tinyimagenet')(input_shape, 
    #                                                  num_classes,
    #                                                  dense_classifier = True).cuda()
    # elif args.arch == 'resnet110' or args.arch == 'resnet110full':
    #     # Using resnet110 from Apollo
    #     # model = apolo_resnet.ResNet(110, num_classes=num_classes)
    #     cutmodel = load.model(args.arch, 'lottery')(input_shape, 
    #                                          num_classes,
    #                                          dense_classifier = True).cuda()
    # elif args.arch in ['vgg16full', 'vgg16full-bn', 'vgg11full', 'vgg11full-bn'] :
    #     if args.dataset == 'tiny-imagenet':
    #         modeltype = 'tinyimagenet'
    #     else:
    #         modeltype = 'lottery'
    #     # Using resnet110 from Apollo
    #     # model = apolo_resnet.ResNet(110, num_classes=num_classes)
    #     cutmodel = load.model(args.arch, modeltype)(input_shape, 
    #                                          num_classes,
    #                                          dense_classifier = True).cuda()
    else:
        raise NotImplementedError('Only ResNet20 can be used for snipres method.')

    # first add masks to each layer of nets
    for net in nets:
        net.train()
        net.zero_grad()
        for layer in net.modules():
            snip.add_mask_ones(layer)
    model = nets[0]

    # add masks to cutmodel
    cutmodel.train()
    cutmodel.zero_grad()
    for layer in cutmodel.modules():
        snip.add_mask_ones(layer)
    
    # data_iter = iter(snip_loader)
    if not args.iter_prune:
        data_iter = iter(data_loader)
        print('[*] Using single-shot SNIP.')
        # Let the neural network run one forward pass to get connect sensitivity (CS)
        for i in range(samples_per_class):
            try:
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            except:
                data_iter = iter(dataloader)
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            # (input, target) = data_iter.next()
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
            if args.half:
                input_var = input_var.half()
            # compute output
            output = cutmodel(input_var)
            loss = criterion(output, target_var)
            loss.backward()
        
        # prune the network using CS
        snip.net_prune_snip(cutmodel, args.sparse_lvl)
        with torch.no_grad():
            for modellayer, cutmodellayer in zip(model.modules(), cutmodel.modules()):
                if isinstance(modellayer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    modellayer.weight = cutmodellayer.weight
                    modellayer.weight_mask = cutmodellayer.weight_mask
        # print('[*] SNIP pruning done!')
        cutmodel = None
        print('[*] SNIP weight pruning done!')

    else:
        print('[*] Using iterative SNIP.')
        num_iter = 10
        data_iter = iter(data_loader)
        for i in range(num_iter):
            try:
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            except:
                data_iter = iter(dataloader)
                (input, target) = snip.GraSP_fetch_data(data_iter, num_classes, samples_per_class)
            # (input, target) = data_iter.next()
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
            if args.half:
                input_var = input_var.half()
            # compute output
            output = cutmodel(input_var)
            loss = criterion(output, target_var)
            if args.ep_coe!=0:
                loss += args.ep_coe * get_ep(model)
            loss.backward()
            # prune the network using CS
            snip.net_iterative_prune(cutmodel, args.sparse_lvl**((i+1)/num_iter))

        with torch.no_grad():
            for modellayer, cutmodellayer in zip(model.modules(), cutmodel.modules()):
                if isinstance(modellayer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    modellayer.weight = cutmodellayer.weight
                    modellayer.weight_mask = cutmodellayer.weight_mask
        # print('[*] SNIP pruning done!')
        cutmodel = None
        print('[*] Iterative SNIP pruning done!')