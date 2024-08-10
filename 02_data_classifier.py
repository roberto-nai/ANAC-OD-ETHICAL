# 02_data_classifier.py

### IMPORT ###
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# ML
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


### LOCAL IMPORT ###
from config import config_reader
from utility_manager.utilities import json_to_list_dict, json_to_sorted_dict, check_and_create_directory,  get_values_from_dict_list, df_read_csv, df_print_details, script_info

### GLOBALS ###
yaml_config = config_reader.config_read_yaml("config.yml", "config")
od_anac_dir = str(yaml_config["OD_ANAC_DIR"]) # input
csv_sep = str(yaml_config["CSV_FILE_SEP"]) # input
conf_file_cols_type = str(yaml_config["CONF_COLS_TYPE_FILE"]) #input

ml_results_dir = str(yaml_config["ML_RESULTS_DIR"])
ml_plots_dir = str(yaml_config["ML_PLOTS_DIR"])

script_path, script_name = script_info(__file__)

ht = 0  # <-- INPUT: set to 1 to apply Hyperopt for hyperparameter tuning
ht_str = ""

# Define hyperparameter search space and optimisation function
def hyperopt_train_test(params, model_class, X_train, y_train):
    # Initialise the model with the current set of parameters
    model = model_class(**params)
    
    # Perform cross-validation and return the mean AUC score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    return auc_scores.mean()

def optimise_model(params, model_class, X_train, y_train):
    # Objective function for Hyperopt
    def objective(params):
        auc_score = hyperopt_train_test(params, model_class, X_train, y_train)
        return {'loss': -auc_score, 'status': STATUS_OK}

    # Perform optimisation
    trials = Trials()
    best_params = fmin(fn=objective, space=params, algo=tpe.suggest, max_evals=50, trials=trials)
    return best_params

### MAIN ###
def main():
    print()
    print(f"*** PROGRAM START ({script_name}) ***")
    print()

    # print(yaml_config) # debug

    start_time = datetime.now().replace(microsecond=0)
    print("Start process: " + str(start_time))
    print()

    print(">> Reading the configuration file")
    print("File (columns type):", conf_file_cols_type)
    list_col_type_dic = json_to_sorted_dict(conf_file_cols_type)
    print()

    print(">> Preparing output directories")
    check_and_create_directory(ml_results_dir)
    check_and_create_directory(ml_plots_dir)
    print()
    
    print(">> Reading input data")
    file_od = "TENDER_NOTICE_filtered.csv"
    print("File:", file_od)
    # Read the file (dataset)
    list_col_exc = ['cig', 'numero_gara', 'oggetto_lotto', 'luogo_istat', 'provincia', 'n_lotti_componenti',  'data_pubblicazione', 'data_scadenza_offerta', 'tipo_scelta_contraente',
    'codice_ausa', 'cf_amministrazione_appaltante', 'denominazione_amministrazione_appaltante', 
    'anno_pubblicazione', 'cod_cpv', 'descrizione_cpv', 'FLAG_PNRR_PNC', 
    'codice_fiscale', 'partita_iva', 'denominazione', 'natura_giuridica_codice', 
    'natura_giuridica_descrizione', 'soggetto_estero', 'provincia_codice', 
    'provincia_nome', 'citta_codice', 'citta_nome', 'indirizzo_odonimo', 
    'cap', 'flag_inHouse', 'flag_partecipata', 'stato', 'data_inizio', 
    'data_fine', 'Codice Regione', 'Codice dell\'Unita territoriale sovracomunale (valida a fini statistici)', 
    'Codice Provincia (Storico)(1)', 'Progressivo del Comune (2)', 'Codice NUTS2 2021 (3) ', 'Codice NUTS2 2024 (3) ',
    'Denominazione (Italiana e straniera)', 'Denominazione in italiano', 
    'Denominazione altra lingua', 'Codice Ripartizione Geografica', 
    'ripartizione_geografica', 'Denominazione Regione', 
    'Denominazione dell\'Unita territoriale sovracomunale (valida a fini statistici)', 
    'Tipologia di Unita territoriale sovracomunale ', 
    'Flag Comune capoluogo di provincia/citta metropolitana/libero consorzio', 
    'Sigla automobilistica', 'Codice Comune formato numerico', 
    'Codice Comune numerico con 110 province (dal 2010 al 2016)', 
    'Codice Comune numerico con 107 province (dal 2006 al 2009)', 
    'Codice Comune numerico con 103 province (dal 1995 al 2005)', 
    'Codice Catastale del comune', 'Codice NUTS1 2021', 'Codice NUTS2 2021 (3)', 
    'Codice NUTS3 2021', 'Codice NUTS1 2024', 'Codice NUTS2 2024 (3)', 
    'Codice NUTS3 2024', 'Codice Istat del Comune (numerico)',
    'Zona altimetrica', 'Altitudine del centro (metri)']
    df_td = df_read_csv(od_anac_dir, file_od, list_col_exc, None, None, csv_sep) # df_td -> dataframe tanders
    df_print_details(df_td, f"File '{file_od}'")
    print()

    print(">> Applying predictions")
    ht_str = "yes" if ht == 1 else "no"
    print("Hypertuning:", ht_str)
    print()

    print("> Preparing data")
    # Select rows with id_variante = 0 and id_variante = 1
    variant_0 = df_td[df_td['id_variante'] == 0]
    variant_1 = df_td[df_td['id_variante'] == 1]

    # Ensure the number of samples in both classes are the same
    n_samples = min(len(variant_0), len(variant_1))
    variant_0_sampled = variant_0.sample(n_samples, random_state=42)
    variant_1_sampled = variant_1.sample(n_samples, random_state=42)

    # Combine samples to create a balanced dataset
    balanced_data = pd.concat([variant_0_sampled, variant_1_sampled])

    print("Balanced data shape:", balanced_data.shape)
    print("Balanced data columns:", balanced_data.columns)
    print()

    # Prepare features and target
    X = balanced_data.drop(columns=['id_variante'])
    y = balanced_data['id_variante'].astype(int)

    # Handle missing values by filling them with the median of each column
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Convert categorical columns to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardise the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialise classifiers and hyperparameter search space
    models = {
        'Logistic Regression': {
            'class': LogisticRegression,
            'params': {'C': hp.loguniform('C', np.log(0.001), np.log(10)), 'max_iter': 1000}
        },
        'Decision Tree': {
            'class': DecisionTreeClassifier,
            'params': {'max_depth': hp.choice('max_depth', range(1, 20)), 'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0)}
        },
        'Random Forest': {
            'class': RandomForestClassifier,
            'params': {'n_estimators': hp.choice('n_estimators', range(10, 500)), 'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])}
        },
        'XGBoost': {
            'class': XGBClassifier,
            'params': {'n_estimators': hp.choice('n_estimators_xgb', range(10, 500)), 'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)}
        },
        'Naive Bayes': {
            'class': GaussianNB,
            'params': {}  # No hyperparameters to tune for GaussianNB
        }
    }
    models_len = len(models)

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("> Training models")
    # Train and evaluate each model with cross-validation
    results = {}
    i = 0
    for model_name, model_info in models.items():
        i += 1
        print(f"[{i}/{models_len}]: {model_name}")

        model_class = model_info['class']
        search_space = model_info['params']

        if ht == 1 and search_space:
            # Apply Hyperopt for hyperparameter tuning
            print(f"Tuning hyperparameters for {model_name}...")
            best_params = optimise_model(search_space, model_class, X_train_scaled, y_train)
            print(f"Best parameters for {model_name}: {best_params}")
        else:
            # Use default hyperparameters
            best_params = {}

        # Initialise model with best or default parameters
        model = model_class(**best_params)

        # Calculate cross-validated scores
        print("> Cross validation")
        cv_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        cv_auc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        cv_precision = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='precision')
        cv_recall = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='recall')

        # Fit model on the entire training set
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'cv_accuracy': np.mean(cv_accuracy),
            'cv_f1': np.mean(cv_f1),
            'cv_auc': np.mean(cv_auc),
            'cv_precision': np.mean(cv_precision),
            'cv_recall': np.mean(cv_recall)
        }
        print()

        print("> Displaying results")
        for model_name, metrics in results.items():
            print(f"Model: {model_name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"AUC: {metrics['auc']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"CV Accuracy: {metrics['cv_accuracy']:.4f}")
            print(f"CV F1 Score: {metrics['cv_f1']:.4f}")
            print(f"CV AUC: {metrics['cv_auc']:.4f}")
            print(f"CV Precision: {metrics['cv_precision']:.4f}")
            print("\n")

    # Prediction concluded
    print("Prediction concluded, showing the results")
    # Convert results to a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Model'
    results_df.reset_index(inplace=True)
    results_df = results_df.sort_values(by='auc', ascending=False)

    # Display the DataFrame
    print(results_df)
    print()

    # Save the ML results
    file_name = f"ml_results_HT_{ht_str}"
    path_ml = Path(ml_results_dir) / f"{file_name}.csv"
    print("Results saved to:", path_ml)
    results_df.to_csv(path_ml, sep=";", index=False)
    path_ml = Path(ml_results_dir) / f"{file_name}.xlsx"
    print("Results saved to:", path_ml)
    results_df.to_excel(path_ml, index=False, sheet_name=file_name)
    print()

    # Program end
    end_time = datetime.now().replace(microsecond=0)
    delta_time = end_time - start_time

    print()
    print("End process:", end_time)
    print("Time to finish:", delta_time)
    print()

    print()
    print("*** PROGRAM END ***")
    print()

if __name__ == "__main__":
    main()