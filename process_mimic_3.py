from modules.mimic_parser import MimicParser


def main():
    ROOT = "./mimic_3_database"
    FOLDER = 'mapped_elements'
    FILE_STR = 'CHARTEVENTS'
    id_col = 'ITEMID'
    label_col = 'LABEL'
    mimic_version = 3

    mp = MimicParser(ROOT, FOLDER, FILE_STR, id_col, label_col, mimic_version)
    mp.perform_full_parsing()


if __name__ == '__main__':
    main()
