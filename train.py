from classes.mimic_parser import MimicParser
from classes.mimic_pre_processor import MimicPreProcessor
from modules.train_model import train_model


def get_targets():
    return ['MI', 'SEPSIS', 'VANCOMYCIN']


def get_percentages():
    return [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]


def pre_process_mimic(filepath, targets):
    mimic_pre_processor = MimicPreProcessor(filepath)
    for target in targets:
        print(f'Creating Datasets for {target}')
        mimic_pre_processor.pre_process_and_save_files(target)
        print(f'Created Datasets for {target}\n')


def train_models(targets, percentages):
    for target in targets:
        print(f'\nTraining {target}')
        for percentage in percentages:
            p = int(percentage * 100)
            model_name = f'mimic_{target}_{p}_percent'
            train_model(model_name=model_name, target=target, n_percentage=percentage)
            print(f'\rFinished training on {percentage * 100}% of data')


def main(parse_mimic, pre_process_data, create_models, mimic_version=4):
    print('Start Program')
    mimic_folder_path = f'./mimic_{mimic_version}_database'
    output_folder = 'mapped_elements'
    file_name = 'CHARTEVENTS'
    id_col = 'ITEMID'
    label_col = 'LABEL'

    mp = MimicParser(mimic_folder_path, output_folder, file_name, id_col, label_col, mimic_version)

    if parse_mimic:
        print('Parse Mimic Data')
        mp.perform_full_parsing()

    if pre_process_data:
        print('Preprocess Data')
        if mp.mimic_version == 3:
            output_filepath = mp.an_path + '.csv'
        else:
            output_filepath = mp.aii_path + '.csv'

        targets = get_targets()
        pre_process_mimic(output_filepath, targets)

    if create_models:
        train_models(get_targets(), get_percentages())


if __name__ == "__main__":
    main(True, True, True, 4)
