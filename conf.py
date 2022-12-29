args = None

SST2Opt = {
    'name': 'sst-2',
    'batch_size': 16,

    'learning_rate': 0.00001,  # initial learning rate # 1e-5
    'weight_decay': 0,  # use default
    'eps': 0.00000001,  # 1e-8

    'file_path': './dataset/sst-2',
    'classes': ['0', '1'],
    'num_class': 2,
    'domains': ["train", "test"],
}