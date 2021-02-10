import csv
import re
from functools import reduce

import numpy as np
import pandas as pd

ROOT = "./mimic_4_database/"


## Utilities ##

def map_dict(elem, dictionary):
    if elem in dictionary:
        return dictionary[elem]
    else:
        return np.nan


## Proper Classes ##

class ParseItemID(object):
    ''' This class builds the dictionaries depending on desired features '''

    def __init__(self):
        self.dictionary = {}

        self.feature_names = ['RBCs', 'WBCs', 'platelets', 'hemoglobin', 'hemocrit',
                              'atypical lymphocytes', 'bands', 'basophils', 'eosinophils', 'neutrophils',
                              'lymphocytes', 'monocytes', 'polymorphonuclear leukocytes',
                              'temperature (F)', 'heart rate', 'respiratory rate', 'systolic', 'diastolic',
                              'pulse oximetry',
                              'troponin', 'HDL', 'LDL', 'BUN', 'INR', 'PTT', 'PT', 'triglycerides', 'creatinine',
                              'glucose', 'sodium', 'potassium', 'chloride', 'bicarbonate',
                              'blood culture', 'urine culture', 'surface culture', 'sputum' +
                              ' culture', 'wound culture', 'Inspired O2 Fraction', 'central venous pressure',
                              'PEEP Set', 'tidal volume', 'anion gap',
                              'daily weight', 'tobacco', 'diabetes', 'history of CV events']

        self.features = ['$^RBC(?! waste)', '$.*wbc(?!.*apache)', '$^platelet(?!.*intake)',
                         '$^hemoglobin', '$hematocrit(?!.*Apache)',
                         'Differential-Atyps', 'Differential-Bands', 'Differential-Basos', 'Differential-Eos',
                         'Differential-Neuts', 'Differential-Lymphs', 'Differential-Monos', 'Differential-Polys',
                         'temperature f', 'heart rate', 'respiratory rate', 'systolic', 'diastolic',
                         'oxymetry(?! )',
                         'troponin', 'HDL', 'LDL', '$^bun(?!.*apache)', 'INR', 'PTT',
                         '$^pt\\b(?!.*splint)(?!.*exp)(?!.*leak)(?!.*family)(?!.*eval)(?!.*insp)(?!.*soft)',
                         'triglyceride', '$.*creatinine(?!.*apache)',
                         '(?<!boost )glucose(?!.*apache).*',
                         '$^sodium(?!.*apache)(?!.*bicarb)(?!.*phos)(?!.*ace)(?!.*chlo)(?!.*citrate)(?!.*bar)(?!.*PO)',
                         '$.*(?<!penicillin G )(?<!urine )potassium(?!.*apache)',
                         '^chloride', 'bicarbonate', 'blood culture', 'urine culture', 'surface culture',
                         'sputum culture', 'wound culture', 'Inspired O2 Fraction', '$Central Venous Pressure(?! )',
                         'PEEP set', 'tidal volume \(set\)', 'anion gap', 'daily weight', 'tobacco', 'diabetes',
                         'CV - past']

        self.patterns = []
        for feature in self.features:
            if '$' not in feature:
                self.patterns.append('.*{0}.*'.format(feature))
            elif '$' in feature:
                self.patterns.append(feature[1::])
            else:
                print(f"no suitable for feature {feature}")

        self.d_items = pd.read_csv(ROOT + 'D_ITEMS.csv', usecols=['itemid', 'label'])
        self.d_items.dropna(how='any', axis=0, inplace=True)

        self.script_features_names = ['epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux',
                                      'asprin', 'ketorolac', 'acetominophen',
                                      'insulin', 'glucagon',
                                      'potassium', 'calcium gluconate',
                                      'fentanyl', 'magensium sulfate',
                                      'D5W', 'dextrose',
                                      'ranitidine', 'ondansetron', 'pantoprazole', 'metoclopramide',
                                      'lisinopril', 'captopril', 'statin',
                                      'hydralazine', 'diltiazem',
                                      'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
                                      'amiodarone', 'digoxin(?!.*fab)',
                                      'clopidogrel', 'nitroprusside', 'nitroglycerin',
                                      'vasopressin', 'hydrochlorothiazide', 'furosemide',
                                      'atropine', 'neostigmine',
                                      'levothyroxine',
                                      'oxycodone', 'hydromorphone', 'fentanyl citrate',
                                      'tacrolimus', 'prednisone',
                                      'phenylephrine', 'norepinephrine',
                                      'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
                                      'diazepam', 'clonazepam',
                                      'propofol', 'zolpidem', 'midazolam',
                                      'albuterol', 'ipratropium',
                                      'diphenhydramine',
                                      '0.9% Sodium Chloride',
                                      'phytonadione',
                                      'metronidazole',
                                      'cefazolin', 'cefepime', 'vancomycin', 'levofloxacin',
                                      'cipfloxacin', 'fluconazole',
                                      'meropenem', 'ceftriaxone', 'piperacillin',
                                      'ampicillin-sulbactam', 'nafcillin', 'oxacillin',
                                      'amoxicillin', 'penicillin', 'SMX-TMP']

        self.script_features = ['epoetin', 'warfarin', 'heparin', 'enoxaparin', 'fondaparinux',
                                'aspirin', 'keterolac', 'acetaminophen',
                                'insulin', 'glucagon',
                                'potassium', 'calcium gluconate',
                                'fentanyl', 'magnesium sulfate',
                                'D5W', 'dextrose',
                                'ranitidine', 'ondansetron', 'pantoprazole', 'metoclopramide',
                                'lisinopril', 'captopril', 'statin',
                                'hydralazine', 'diltiazem',
                                'carvedilol', 'metoprolol', 'labetalol', 'atenolol',
                                'amiodarone', 'digoxin(?!.*fab)',
                                'clopidogrel', 'nitroprusside', 'nitroglycerin',
                                'vasopressin', 'hydrochlorothiazide', 'furosemide',
                                'atropine', 'neostigmine',
                                'levothyroxine',
                                'oxycodone', 'hydromorphone', 'fentanyl citrate',
                                'tacrolimus', 'prednisone',
                                'phenylephrine', 'norepinephrine',
                                'haloperidol', 'phenytoin', 'trazodone', 'levetiracetam',
                                'diazepam', 'clonazepam',
                                'propofol', 'zolpidem', 'midazolam',
                                'albuterol', '^ipratropium',
                                'diphenhydramine(?!.*%)(?!.*cream)(?!.*/)',
                                '^0.9% sodium chloride(?! )',
                                'phytonadione',
                                'metronidazole(?!.*%)(?! desensit)',
                                'cefazolin(?! )', 'cefepime(?! )', 'vancomycin', 'levofloxacin',
                                'cipfloxacin(?!.*ophth)', 'fluconazole(?! desensit)',
                                'meropenem(?! )', 'ceftriaxone(?! desensit)', 'piperacillin',
                                'ampicillin-sulbactam', 'nafcillin', 'oxacillin', 'amoxicillin',
                                'penicillin(?!.*Desen)', 'sulfamethoxazole']

        self.script_patterns = ['.*' + feature + '.*' for feature in self.script_features]

    def prescriptions_init(self):
        self.prescriptions = pd.read_csv(ROOT + 'PRESCRIPTIONS.csv',
                                         usecols=['subject_id', 'hadm_id', 'drug',
                                                  'starttime', 'stoptime'])
        self.prescriptions.dropna(how='any', axis=0, inplace=True)

    def extractor(self, feature_name, pattern):
        condition = self.d_items['label'].str.contains(pattern, flags=re.IGNORECASE)
        dictionary_value = self.d_items['itemid'].where(condition).dropna().values.astype('int')
        self.dictionary[feature_name] = set(dictionary_value)

    def build_dictionary(self):
        assert len(self.feature_names) == len(self.features)
        for feature, pattern in zip(self.feature_names, self.patterns):
            self.extractor(feature, pattern)

    def reverse_dictionary(self, dictionary):
        self.rev = {}
        for key, value in dictionary.items():
            for elem in value:
                self.rev[elem] = key


class MimicParser(object):
    ''' This class structures the MIMIC III and builds features then makes 24 hour windows '''

    def __init__(self):
        self.name = 'mimic_assembler'
        self.pid = ParseItemID()
        self.pid.build_dictionary()
        self.features = self.pid.features

    def reduce_total(self, filepath):

        ''' This will filter out rows from CHARTEVENTS.csv that are not feature relevant '''

        # CHARTEVENTS = 330712484

        pid = ParseItemID()
        pid.build_dictionary()
        chunksize = 10000000
        columns = ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value', 'valuenum']

        for i, df_chunk in enumerate(pd.read_csv(filepath, iterator=True, chunksize=chunksize)):
            function = lambda x, y: x.union(y)
            df = df_chunk[df_chunk['itemid'].isin(reduce(function, pid.dictionary.values()))]
            df.dropna(inplace=True, axis=0, subset=columns)
            if i == 0:
                df.to_csv(ROOT + './mapped_elements/CHARTEVENTS_reduced.csv', index=False,
                          columns=columns)
            else:
                df.to_csv(ROOT + './mapped_elements/CHARTEVENTS_reduced.csv', index=False,
                          columns=columns, header=None, mode='a')
            print(i)

    def create_day_blocks(self, file_name):

        ''' Uses pandas to take shards and build them out '''
        pid = ParseItemID()
        pid.build_dictionary()
        pid.reverse_dictionary(pid.dictionary)
        df = pd.read_csv(file_name)
        print("loaded df")
        df['chartday'] = df['charttime'].astype('str').str.split(' ').apply(lambda x: x[0])
        print("New feature chartday")
        df['hadmid_day'] = df['hadm_id'].astype('str') + '_' + df['chartday']
        print("New feature hadmid_day")
        df['features'] = df['itemid'].apply(lambda x: pid.rev[x])
        print("New feature features")

        self.hadm_dict = dict(zip(df['hadmid_day'], df['subject_id']))
        print("start statistics")
        df2 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', fill_value=np.nan)
        df3 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', aggfunc=lambda x: np.std(x), fill_value=0)
        df3.columns = ["{0}_std".format(i) for i in list(df2.columns)]
        print("std finished")
        df4 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', aggfunc=np.amin, fill_value=np.nan)
        df4.columns = ["{0}_min".format(i) for i in list(df2.columns)]
        print("min finished")
        df5 = pd.pivot_table(df, index='hadmid_day', columns='features',
                             values='valuenum', aggfunc=np.amax, fill_value=np.nan)
        df5.columns = ["{0}_max".format(i) for i in list(df2.columns)]
        print("max finished")
        df2 = pd.concat([df2, df3, df4, df5], axis=1)

        df2['tobacco'].apply(lambda x: np.around(x))
        del df2['daily weight_std']
        del df2['daily weight_min']
        del df2['daily weight_max']
        del df2['tobacco_std']
        del df2['tobacco_min']
        del df2['tobacco_max']

        rel_columns = [i for i in list(df2.columns) if '_' not in i]

        for col in rel_columns:
            if len(np.unique(df2[col])[np.isfinite(np.unique(df2[col]))]) <= 2:
                print(col)
                del df2[col + '_std']
                del df2[col + '_min']
                del df2[col + '_max']

        for i in list(df2.columns):
            df2[i][df2[i] > df2[i].quantile(.95)] = df2[i].median()
            df2[i].fillna(df2[i].median(), inplace=True)

        df2['hadmid_day'] = df2.index
        df2.dropna(thresh=int(0.75 * len(df2.columns)), axis=0, inplace=True)
        df2.to_csv(file_name[0:-4] + '_24_hour_blocks.csv', index=False)

    def add_admissions_columns(self, file_name):

        ''' Add demographic columns to create_day_blocks '''

        df = pd.read_csv(f'{ROOT}/ADMISSIONS.csv')
        ethn_dict = dict(zip(df['hadm_id'], df['ethnicity']))
        admittime_dict = dict(zip(df['hadm_id'], df['admittime']))
        df_shard = pd.read_csv(file_name)
        df_shard['hadm_id'] = df_shard['hadmid_day'].str.split('_').apply(lambda x: x[0])
        df_shard['hadm_id'] = df_shard['hadm_id'].astype('int')
        df_shard['ethnicity'] = df_shard['hadm_id'].apply(lambda x: map_dict(x, ethn_dict))
        black_condition = df_shard['ethnicity'].str.contains('.*black.*', flags=re.IGNORECASE)
        df_shard['black'] = 0
        df_shard['black'][black_condition] = 1
        del df_shard['ethnicity']
        df_shard['admittime'] = df_shard['hadm_id'].apply(lambda x: map_dict(x, admittime_dict))
        df_shard.to_csv(file_name[0:-4] + '_plus_admissions.csv', index=False)

    def add_patient_columns(self, file_name):

        ''' Add demographic columns to create_day_blocks '''

        df = pd.read_csv(f'{ROOT}/PATIENTS.csv')

        # Date of birth omitted in MIMIC 4
        gender_dict = dict(zip(df['subject_id'], df['gender']))
        df_shard = pd.read_csv(file_name)
        df_shard['subject_id'] = df_shard['hadmid_day'].apply(lambda x:
                                                              map_dict(x, self.hadm_dict))
        df_shard['admityear'] = df_shard['admittime'].str.split('-').apply(lambda x: x[0]).astype('int')
        df_shard['gender'] = df_shard['subject_id'].apply(lambda x: map_dict(x, gender_dict))
        gender_dummied = pd.get_dummies(df_shard['gender'], drop_first=True)
        gender_dummied.rename(columns={'M': 'Male', 'F': 'Female'})
        COLUMNS = list(df_shard.columns)
        COLUMNS.remove('gender')
        df_shard = pd.concat([df_shard[COLUMNS], gender_dummied], axis=1)
        df_shard.to_csv(file_name[0:-4] + '_plus_patients.csv', index=False)

    def clean_prescriptions(self):

        ''' Add prescriptions '''

        pid = ParseItemID()
        pid.prescriptions_init()
        pid.prescriptions.drop_duplicates(inplace=True)
        pid.prescriptions['drug_feature'] = np.nan
        print(f"{len(pid.script_features_names)}")
        count = 1
        for feature, pattern in zip(pid.script_features_names, pid.script_patterns):
            print(f"\r{count}-{feature}", end=" ")
            condition = pid.prescriptions['drug'].str.contains(pattern, flags=re.IGNORECASE)
            pid.prescriptions['drug_feature'][condition] = feature
            count += 1
        print("\rLeft loop")
        pid.prescriptions.dropna(how='any', axis=0, inplace=True, subset=['drug_feature'])

        pid.prescriptions.to_csv(f'{ROOT}/PRESCRIPTIONS_reduced.csv', index=False)

    def add_prescriptions(self, file_name):

        df_file = pd.read_csv(file_name)

        with open(f'{ROOT}/PRESCRIPTIONS_reduced.csv', 'r') as f:
            csvreader = csv.reader(f)
            print(sum(1 for row in csvreader))
        with open(f'{ROOT}/PRESCRIPTIONS_reduced.csv', 'r') as f:
            csvreader = csv.reader(f)
            with open(f'{ROOT}/PRESCRIPTIONS_reduced_byday.csv', 'w') as g:
                csvwriter = csv.writer(g)
                first_line = csvreader.__next__()
                print(first_line[0:2] + ['chartday'] + [first_line[-1]])
                csvwriter.writerow(first_line[0:2] + ['chartday'] + [first_line[-1]])
                count = 1
                for row in csvreader:
                    for i in pd.date_range(row[2], row[3]).strftime('%Y-%m-%d'):
                        csvwriter.writerow(row[0:2] + [i] + [row[-1]])
                    print(f"\r{count}", end="")
                    count += 1
        print("\rFinished Loop")
        df = pd.read_csv(f'{ROOT}/PRESCRIPTIONS_reduced_byday.csv')
        df['chartday'] = df['chartday'].str.split(' ').apply(lambda x: x[0])
        df['hadmid_day'] = df['hadm_id'].astype('str') + '_' + df['chartday']
        df['value'] = 1

        cols = ['hadmid_day', 'drug_feature', 'value']
        df = df[cols]

        df_pivot = pd.pivot_table(df, index='hadmid_day', columns='drug_feature', values='value', fill_value=0,
                                  aggfunc=np.amax)
        df_pivot.reset_index(inplace=True)

        df_merged = pd.merge(df_file, df_pivot, on='hadmid_day', how='outer')

        del df_merged['hadm_id']
        df_merged['hadm_id'] = df_merged['hadmid_day'].str.split('_').apply(lambda x: x[0])
        df_merged.fillna(0, inplace=True)

        df_merged['dextrose'] = df_merged['dextrose'] + df_merged['D5W']
        del df_merged['D5W']

        df_merged.to_csv(file_name[0:-4] + '_plus_scripts.csv', index=False)

    def add_icd_infect(self, file_name):

        df_icd = pd.read_csv(f'{ROOT}/PROCEDURES_ICD.csv')
        df_micro = pd.read_csv(f'{ROOT}/MICROBIOLOGYEVENTS.csv')
        self.suspect_hadmid = set(pd.unique(df_micro['hadm_id']).tolist())
        df_icd_ckd = df_icd[df_icd['icd_code'] == 585]

        self.ckd = set(df_icd_ckd['hadm_id'].values.tolist())

        df = pd.read_csv(file_name)
        df['ckd'] = df['hadm_id'].apply(lambda x: 1 if x in self.ckd else 0)
        df['infection'] = df['hadm_id'].apply(lambda x: 1 if x in self.suspect_hadmid else 0)
        df.to_csv(file_name[0:-4] + '_plus_icds.csv', index=False)

    def add_notes(self, file_name):
        df = pd.read_csv(f'{ROOT}/NOTEEVENTS.csv')
        df_rad_notes = df[['text', 'hadm_id']][df['category'] == 'Radiology']
        CTA_bool_array = df_rad_notes['text'].str.contains('CTA', flags=re.IGNORECASE)
        CT_angiogram_bool_array = df_rad_notes['text'].str.contains('CT angiogram', flags=re.IGNORECASE)
        chest_angiogram_bool_array = df_rad_notes['text'].str.contains('chest angiogram', flags=re.IGNORECASE)
        cta_hadm_ids = np.unique(df_rad_notes['hadm_id'][CTA_bool_array].dropna())
        CT_angiogram_hadm_ids = np.unique(df_rad_notes['hadm_id'][CT_angiogram_bool_array].dropna())
        chest_angiogram_hadm_ids = np.unique(df_rad_notes['hadm_id'][chest_angiogram_bool_array].dropna())
        hadm_id_set = set(cta_hadm_ids.tolist())
        hadm_id_set.update(CT_angiogram_hadm_ids)
        print(len(hadm_id_set))
        hadm_id_set.update(chest_angiogram_hadm_ids)
        print(len(hadm_id_set))

        df2 = pd.read_csv(file_name)
        df2['ct_angio'] = df2['hadm_id'].apply(lambda x: 1 if x in hadm_id_set else 0)
        df2.to_csv(file_name[0:-4] + '_plus_notes.csv', index=False)


def main():
    pid = ParseItemID()
    pid.build_dictionary()
    FOLDER = 'mapped_elements/'
    FILE_STR = 'CHARTEVENTS_reduced'
    mp = MimicParser()

    #mp.reduce_total(ROOT + 'CHARTEVENTS.csv')
    print("Created Chartevents")
    #mp.create_day_blocks(ROOT + FOLDER + FILE_STR + '.csv')
    print("Created Dayblocks")
    #mp.add_admissions_columns(ROOT + FOLDER + FILE_STR + '_24_hour_blocks.csv')
    print("Created 24h blocks")
    #mp.add_patient_columns(ROOT + FOLDER + FILE_STR + '_24_hour_blocks_plus_admissions.csv')
    print("Created Plus Admission")
    #mp.clean_prescriptions()
    print("Cleaned Plus Patients")
    #mp.add_prescriptions(ROOT + FOLDER + FILE_STR + '_24_hour_blocks_plus_admissions_plus_patients.csv')
    print("Created Plus Patients")
    #mp.add_icd_infect(ROOT + FOLDER + FILE_STR + '_24_hour_blocks_plus_admissions_plus_patients_plus_scripts.csv')
    print("Created Plus Scripts")
    mp.add_notes(ROOT + FOLDER + FILE_STR + '_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds.csv')
    print("Created Plus ICDS")


if __name__ == '__main__':
    main()
