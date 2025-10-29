from common.attribute_dict import AttrDict

CFG = dict(
    data_dir=r"./data/test_colmap",
    save_dir=r"./output",
    data_device="cuda:0",
    resolution=-1,
    sh_degrees=3,
    train_test_exp=False,
    resolution_scale=1.0,
    white_background=True,
    random_background=False,
    debug=False,
    antialiasing=False,
    compute_cov3D_python=False,
    convert_SHs_python=False,

    # training parameters
    first_iter=0,
    iterations=30_000,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30_000,
    feature_lr=0.0025,
    opacity_lr=0.025,
    scaling_lr=0.005,
    rotation_lr=0.001,
    exposure_lr_init=0.01,
    exposure_lr_final=0.001,
    exposure_lr_delay_steps=0,
    exposure_lr_delay_mult=0.0,
    percent_dense=0.01,
    lambda_dssim=0.2,
    densification_interval=100,
    opacity_reset_interval=3000,
    densify_from_iter=500,
    densify_until_iter=15_000,
    densify_grad_threshold=0.0002,
    depth_l1_weight_init=1.0,
    depth_l1_weight_final=0.01,
    optimizer_type="default"
)

CFG = AttrDict(CFG)
