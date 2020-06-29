import fire

from MSRResNet import test


def test_prune_test():
    # Initialize variables
    data_dir='../dataset'
    root_dir='MSRResNet'
    save=False
    # --------------------------------
    # basic settings
    # --------------------------------
    testsets = f'{data_dir}/DIV2K'
    testset_L = f'DIV2K_valid_LR_bicubic'
    testset_H = f'DIV2K_valid_HR'
    L_folder = os.path.join(testsets, testset_L, 'X4')
    H_folder = os.path.join(testsets, testset_H)
    E_folder = os.path.join('results')
    P_folder = os.path.join('pruned')
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger('AIM-track')
    model_path = os.path.join(root_dir, 'MSRResNetx4_model', 'MSRResNetx4.pth')
    # Load model
    model = load_model(model_path, device)
    logger.info(f'Params number: {count_num_of_parameters(model)}')
    # test
    test(model, L_folder, E_folder, logger, True)
    logger.info(f'PSNR before pruning {calculate_psnr(H_folder, E_folder, logger)}')
    # prune
    import copy

    model_new = copy.deepcopy(model)
    logger.info('Params number(Before prune): {}'.format(count_num_of_parameters(model_new)))
    pre_mask_index = torch.ones(3, dtype=torch.bool).to(device)
    for name, module in model_new.named_modules():
        if 'conv' in name:
            if name in ['upconv1', 'upconv2']:
                continue
            prune.ln_structured(module, 'weight', amount=0.5, n=2, dim=0)
            prune.remove(module, 'weight')
            mask_index = module.weight.sum(-1).sum(-1).sum(-1) != 0
            # DEBUG ----------------------------------------------------
            logger.debug("=" * 20)
            logger.debug(f"{name}: Pruned")
            logger.debug(f"pre_mask_index.shape: {pre_mask_index.shape}")
            logger.debug(f"mask_index.shape: {mask_index.shape}")
            pre_module_weight_shape = module.weight.shape
            pre_module_bias_shape = module.bias.shape
            # DEBUG ----------------------------------------------------
            if name not in ['conv_first'] + [f"recon_trunk.{i}.conv1" for i in range(16)] + ['HRconv']:
                module.weight = torch.nn.Parameter(module.weight[:, pre_mask_index])
            if name not in ['conv_first'] + [f"recon_trunk.{i}.conv2" for i in range(16)] + ['conv_last']:
                module.weight = torch.nn.Parameter(module.weight[mask_index, :])
                module.bias = torch.nn.Parameter(module.bias[mask_index])
            # DEBUG ----------------------------------------------------
            logger.debug(f"module.weight.shape: {pre_module_weight_shape} --> {module.weight.shape}")
            logger.debug(f"module.bias.shape: {pre_module_bias_shape} --> {module.bias.shape}")
            # DEBUG ----------------------------------------------------
            pre_mask_index = mask_index
        else:
            logger.debug("=" * 20)
            logger.debug(f"{name}: Unpruned")
    logger.info('Params number(After prune): {}'.format(count_num_of_parameters(model_new)))
    # test
    test(model_new, L_folder, P_folder, logger, True)
    logger.info(f'PSNR after pruning {calculate_psnr(H_folder, P_folder, logger)}')


def inference(image_path):
    root_dir='MSRResNet'
    model_path = os.path.join(root_dir, 'MSRResNetx4_model', 'MSRResNetx4.pth')
    model = load_model(model_path, device)
    model = load_model(model_path, device)



def hello():
    print("Hello, World!")


if __name__ == '__main__':
    fire.Fire({
        'test': test,
        'hello': hello
    })