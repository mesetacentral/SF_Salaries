import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

job_classification = {'FIRE': ['FIRE'],
                     'POLICE': ['POLICE', 'SHERIF', 'PROBATION', 'SERGEANT', 'FORENSIC'],
                     'TRANSIT': ['TRANSIT', 'MTA', 'TRANSPORT', 'TRAF'],
                     'MEDICAL': ['MEDICAL', 'ANESTH',  'NURS', 'HEALTH', 'ORTHOPEDIC', 'PHYSICIAN', 'HEALTH', 'PHARM', 'DIAGNOSTIC', 'PSYCH', 'SURG', 'HOSPITAL', 'DENTAL', 'THERAP', 'EPIDEM', 'DISEASE', 'BACTER'],
                     'AIRPORT': ['AIRPORT'],
                     'ANIMAL': ['ANIMAL'],
                     'ARCHITECTURAL': ['ARCHITECT'],
                     'COURT': ['COURT', 'LEGAL', 'COUNSELOR', 'LAW', 'FISCAL'],
                     'MAYOR': ['MAYOR'],
                     'LIBRARY': ['LIBRAR'],
                     'PARKING': ['PARKING'],
                     'PUBLIC WORKS': ['PUBLIC WORKS', 'BUILD', 'CEMENT', 'BRICK'],
                     'ATTORNEY': ['ATTORNEY'],
                     'AUTOMOTIVE': ['AUTOMOTIVE', 'CAR '],
                     'CUSTODIAN': ['CUSTODIAN'],
                     'ENGINEER': ['ENGINEER', 'ENGR', 'ENG'],
                     'ACCOUNTING': ['ACCOUNT'],
                     'GARDENER': ['GARDEN', 'TREE', 'FOREST'],
                     'GENERAL LABORER': ['GENERAL LABORER', 'PLUM ', 'MECHANIC', 'ELECTRICIAN', 'REPAIRER', 'PAINTER', 'CARPENTER', 'CLEANER'],
                     'FOOD SERVICE': ['FOOD SERV'],
                     'CLERK': ['CLERK'],
                     'PORTER': ['PORTER'],
                     'PORT': ['PORT '],
                     'GUARD': ['GUARD'],
                     'HUMAN RESOURCES': ['HUMAN']}

ranking_classification = {'LOW': [''],
                          'HIGH': ['SENIOR', 'MANAGER', 'III', 'IV', ' V', 'VI', 'CHIEF', 'HEAD', 'DIRECTOR', 'SUPERVISOR', 'CAPTAIN', 'LIEUTENANT', 'SERGEANT']}

path = '../test/'

def add_classification(df, column, new_column, keywords):
    for classification in keywords.keys():
        for keyword in keywords[classification]:
            df.loc[df[column].str.contains(keyword), new_column] = classification

def generate_features():
    # Data clean up
    df = pd.read_csv(r"../data/Salaries.csv", header=0, delimiter=',')
    
    df = df.drop(['Notes', 'Agency', 'Id', 'Status'], axis=1)
    
    df = df[df.BasePay.notnull()]
    df = df[df.BasePay.ne('Not Provided')]
    
    df.BasePay = df.BasePay.astype('float64')
    df.OvertimePay = df.OvertimePay.astype('float64')
    df.OtherPay = df.OtherPay.astype('float64')
    df.Benefits = df.Benefits.astype('float64')
    
    df.loc[df.Benefits.isnull(), 'Benefits'] = 0
    
    df = df[df.BasePay.ge(0) * df.OvertimePay.ge(0) * df.OtherPay.ge(0) * df.Benefits.ge(0) * df.TotalPay.ge(0) * df.TotalPayBenefits.ge(0)]
    
    df = df[df.iloc[:, np.array([df.dtypes == 'float64']).reshape(df.shape[1], )].ge(0).all(axis=1)]
    df = df.reset_index(drop=True)
    
    df.JobTitle = df.JobTitle.str.upper()
    df.JobTitle = df.JobTitle.replace(to_replace=r'  +', value=' ', regex=True)
    df.EmployeeName = df.EmployeeName.str.upper()
    df.EmployeeName = df.EmployeeName.replace(to_replace=r'  +', value=' ', regex=True)
    
    # Feature extraction
    # Salary
    df = df[df.BasePay > 2.1e4]
    
    # Jobs
    unique_jobs, count_jobs = np.unique(df.JobTitle, return_counts=True)
    sorted_count_jobs = np.array(sorted(count_jobs, reverse=True))
    max_jobs = sorted_count_jobs[sorted_count_jobs > 200].shape[0]
    df = df[df.JobTitle.isin(unique_jobs[np.argsort(count_jobs)][::-1][:max_jobs])].reset_index(drop=True)
    
    add_classification(df, 'JobTitle', 'JobArea', job_classification)
    df = df[df.JobArea.notna()]
    
    # Job Area
    womens_names = pd.read_csv(r"../data/babies-first-names-top-100-girls.csv", header=0, delimiter=',')
    mens_names = pd.read_csv(r"../data/babies-first-names-top-100-boys.csv", header=0, delimiter=',')
    
    womens_names.FirstForename = womens_names.FirstForename.str.upper()
    mens_names.FirstForename = mens_names.FirstForename.str.upper()
    
    gender_classification = {'F': womens_names.FirstForename.to_list(), 
                             'M': mens_names.FirstForename.to_list()}
    
    df.loc[:, 'FirstName'] = df.EmployeeName.apply(lambda x: ''.join(x.split()[0]))
    add_classification(df, 'FirstName', 'Gender', gender_classification)
    
    df = pd.concat([df.loc[df.Gender.eq('M')][:df.Gender.eq('F').sum()], df.loc[df.Gender.eq('F')]]).reset_index(drop=True)
    
    # Ranking
    add_classification(df, 'JobTitle', 'Ranking', ranking_classification)
    
    # Splitting data and tfidf transformation
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, ['JobTitle', 'JobArea', 'Gender', 'Ranking']], df.loc[:, 'TotalPay'])
    
    tfidf = TfidfVectorizer(stop_words='english')
    X_train = tfidf.fit_transform(X_train.JobTitle)
    X_test = tfidf.transform(X_test.JobTitle)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.toarray())
    X_test = scaler.transform(X_test.toarray())
    
    scaler = StandardScaler()
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()
    y_test = scaler.transform(np.array(y_test).reshape(-1, 1)).flatten()
    
    # Saving data
    np.save(os.path.join(path, 'X_train'), X_train)
    np.save(os.path.join(path, 'X_test'), X_test)    
    np.save(os.path.join(path, 'y_train'), y_train)    
    np.save(os.path.join(path, 'y_test'), y_test)
    
    return 0

generate_features()
    
    
    
    
    