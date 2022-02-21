from main_ours_resnet18_msc import *

def main():
    config.backbone = 'resnet50'
    config.pretrained = 'models/imagenet/resnet50-19c8e357.pth'
    config.batch_size = 12 * len(config.gpus)

    set_seed(args.seed)
    config.train_data_dir = args.train_data_dir
    config.train_json_file = args.train_json_file
    config.valid_data_dir = args.valid_data_dir
    config.valid_json_file = args.valid_json_file

    config.output_dir = os.path.join('output', 'ours_resnet50_msc', str(args.fold_id))
    os.makedirs(config.output_dir, exist_ok=True)
    config.logger = build_logging(os.path.join(config.output_dir, 'log.log'))
    
    (train_studies, train_annotation), \
    (valid_studies, valid_annotation) = prepare_data()
    
    train_valid(config, 0, train_studies, train_annotation, valid_studies, valid_annotation)
    
if __name__ == '__main__':
    main()