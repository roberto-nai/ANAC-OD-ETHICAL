# 01_data_analyser.py

### IMPORT ###
import pandas as pd
from datetime import datetime
from pathlib import Path
import csv
import geopandas as gpd
import matplotlib.pyplot as plt
import json

### LOCAL IMPORT ###
from config import config_reader
from utility_manager.utilities import json_to_list_dict, json_to_sorted_dict, check_and_create_directory, list_files_by_type, get_values_from_dict_list, df_read_csv, df_print_details, distinct_values_frequencies, save_stats, script_info, clean_dataframe

### GLOBALS ###
yaml_config = config_reader.config_read_yaml("config.yml", "config")
# print(yaml_config) # debug
od_anac_dir = str(yaml_config["OD_ANAC_DIR"])
od_file_type = str(yaml_config["OD_FILE_TYPE"])
csv_sep = str(yaml_config["CSV_FILE_SEP"])
tender_main_file = str(yaml_config["TENDER_MAIN_FILE"])
conf_file_cols_type = str(yaml_config["CONF_COLS_TYPE_FILE"]) 
conf_file_stats_inc = str(yaml_config["CONF_COLS_STATS_FILE"]) 
conf_file_filters = str(yaml_config["CONF_COLS_FILTER_FILE"]) 

stats_do = 1 # 0 if stats are not needed, else 1

stats_dir =  str(yaml_config["OD_STATS_DIR"])
plot_dir =  str(yaml_config["OD_PLOTS_DIR"])

script_path, script_name = script_info(__file__)

### FUNCTIONS ###

def summarize_dataframe_to_dict(df: pd.DataFrame, file_name: str) -> dict:
    """
    Creates a dictionary summarizing the input DataFrame with the file name, and the count of missing (empty) values for each column.

    Parameters:
        df (pd.DataFrame): The DataFrame to summarize.
        file_name (str): The name of the file associated with the DataFrame.

    Returns:
        dict: a dictionary containing the file name and missing value counts for each column.
    """
    # Count the number of missing values in each column of the DataFrame
    missing_counts = df.isnull().sum()
    # Convert the Series to a dictionary
    missing_counts_dict = missing_counts.to_dict()
    # Count the number of duplicate rows, considering all columns
    duplicate_rows_count = df.duplicated().sum()
    # Get the number of rows and columns in the DataFrame
    num_rows, num_columns = df.shape
    # Calculate the ratio of duplicate rows to total rows
    ratio_dup = duplicate_rows_count / num_rows if num_rows > 0 else 0  # Avoid division by zero

    # Create the summary dictionary
    summary_dict = {
        'file_name': file_name,
        'rows_num':num_rows,
        'cols_num':num_columns,
        'missing_values': missing_counts_dict,
        'duplicated_rows': duplicate_rows_count,
        'duplicated_rows_perc': round(ratio_dup,2)
    }
    return summary_dict

def summarize_dataframe_to_df(summary_dict:dict) -> pd.DataFrame:
    """
    Saves the given summary dictionary to a CSV file, where each key-value pair in the dictionary becomes a column. The 'Missing Values Per Column' nested dictionary is expanded into separate columns.

    Parameters:
        summary_dict (dict): The summary dictionary to save.
        csv_file_name (str): The file name for the CSV file.

    Returns:
        pd.DataFrame: A dataframe with data.
    """
    # Flatten the 'Missing Values Per Column' dictionary into the main dictionary with prefix
    for key, value in summary_dict['missing_values'].items():
        summary_dict[f'Missing_{key}'] = value
    # Remove the original nested dictionary key
    del summary_dict['missing_values']
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([summary_dict])
    # Save the DataFrame to a CSV file
    # df.to_csv(csv_file_name, index=False)
    return df

def plot_italy_region_map(dataframe:pd.DataFrame, region_column:str, amount_column:str, output_folder:str, output_filename:str='italy_region_map'):
    """
    Function to plot a map of Italy coloured by regional amounts and save it to a file. Also saves the aggregated dataframe as a CSV file.
    
    Parameters:
    dataframe (pd.DataFrame): The dataframe containing regions and amounts.
    region_column (str): The name of the column with region names.
    amount_column (str): The name of the column with amounts to be summed and linked to regions.
    output_folder (str): The folder where the output file will be saved.
    output_filename (str): The name of the output file (default is 'italy_region_map.png').

    """
    output_path_csv = Path(output_folder) / f"{output_filename}.csv"
    output_path_png = Path(output_folder) / f"{output_filename}.png"

    # Aggregate the amount_column by region
    df_aggregated = dataframe.groupby(region_column)[amount_column].sum().reset_index()
    df_aggregated.to_csv(output_path_csv, sep=";", index=False)

    # Load the shapefile of Italian regions
    # https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson
    italy_regions = gpd.read_file("limits_IT_regions.geojson")
    
    # Merge the aggregated data with the geographic dataframe
    italy_regions = italy_regions.merge(df_aggregated, how='left', left_on='reg_name', right_on=region_column)
    
    # Create the map
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    italy_regions.plot(column=amount_column, cmap='OrRd', linewidth=1, ax=ax, edgecolor='0.9', legend=False)
    
    # Add titles and labels
    ax.set_title('Amounts by Region in Italy', fontdict={'fontsize': '15', 'fontweight' : '3'})
    ax.set_axis_off()
    
    # Save the plot to the specified file
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close()

    print("Map saved to:", output_path_png)

    plt.show()

def capitalize_region(region):
    if region.lower() == 'friuli venezia giulia':
        return 'Friuli-Venezia Giulia'
    if region.lower() == 'pa bolzano':
        return 'Trentino-Alto Adige/Südtirol'
    if region.lower() == 'pa trento':
        return 'Trentino-Alto Adige/Südtirol'
    if region.lower() == 'centrale':
        return 'Lazio'
    if region.lower() == 'emilia romagna':
        return 'Emilia-Romagna'
    if region.lower() == 'valle d\'aosta':
        return 'Valle d\'Aosta/Vallée d\'Aoste'
    else:
        return region.capitalize()
    

### MAIN ###
def main():
    print()
    print(f"*** PROGRAM START ({script_name}) ***")
    print()

    start_time = datetime.now().replace(microsecond=0)
    print("Start process: " + str(start_time))
    print()

    print(">> Preparing output directories")
    check_and_create_directory(stats_dir)
    check_and_create_directory(plot_dir)
    print()

    print(">> Scanning Open Data catalogue")
    print("Directory:", od_anac_dir)
    list_od_files = list_files_by_type(od_anac_dir, od_file_type)
    list_od_files_len = len(list_od_files)
    print(f"Files '{od_file_type}' found: {list_od_files_len}")
    print()

    print(">> Reading the configuration file")
    
    print("File (columns type):", conf_file_cols_type)
    list_col_type_dic = json_to_sorted_dict(conf_file_cols_type)
    # print(list_col_type_dic) # debug
    
    print("File (stats columns):", conf_file_stats_inc)
    list_col_stats_dic = json_to_list_dict(conf_file_stats_inc)
    # print(list_col_stats_dic) # debug

    print("File (filter columns):", conf_file_filters)
    list_col_filters_dic = json_to_list_dict(conf_file_filters)
    # print(list_col_filters_dic) # debug

    print()

    print(">> Reading Open Data files")

    print("> Reading file")

    # CA
    file_od = "CONTRACTING_AUTHORITIES.csv"
    print("File:", file_od)
    file_path = Path(file_od)
    file_stem = file_path.stem # get the name without extension (is also the event name)

    # Read the file (dataset)
    list_col_exc = [] # no columns to exclude
    df_ca = df_read_csv(od_anac_dir, file_od, list_col_exc, list_col_type_dic, None, csv_sep) # df_td -> dataframe contracting authorities
    df_print_details(df_ca, f"File '{file_od}'")
    print()

    # ISTAT (1)
    file_od = "Codici-statistici-e-denominazioni-al-22_01_2024.csv"
    print("File:", file_od)
    file_path = Path(file_od)
    file_stem = file_path.stem # get the name without extension (is also the event name)

    list_col_exc = [] # no columns to exclude
    df_istat_1 = df_read_csv(od_anac_dir, file_od, list_col_exc, list_col_type_dic, None, csv_sep) # df_td -> dataframe contracting authorities
    df_print_details(df_istat_1, f"File '{file_od}'")
    print()

    # ISTAT (2)
    file_od = "Classificazioni statistiche-e-dimensione-dei-comuni_31_12_2022.csv"
    print("File:", file_od)
    file_path = Path(file_od)
    file_stem = file_path.stem # get the name without extension (is also the event name)

    list_col_exc = [] # no columns to exclude
    df_istat_2 = df_read_csv(od_anac_dir, file_od, list_col_exc, list_col_type_dic, None, csv_sep) # df_td -> dataframe contracting authorities
    df_print_details(df_istat_2, f"File '{file_od}'")
    print()

    # Unique ISTAT (joint)
    df_istat = pd.merge(left=df_istat_1, right=df_istat_2, on="citta_codice", how="left")
    df_istat = clean_dataframe(df_istat)
    df_print_details(df_istat, f"Unique ISTAT dataframe")
    print()

    # Join ISTAT and CA
    df_ca_istat = pd.merge(left=df_ca, right=df_istat, on="citta_codice", how="left")
    df_print_details(df_ca_istat, f"File 'CA x ISTAT'")
    print()

    # AWARD
    file_od = "AWARDS.csv"
    print("File:", file_od)
    file_path = Path(file_od)
    file_stem = file_path.stem # get the name without extension (is also the event name)

    # Read the file (dataset)
    list_col_exc = [] # no columns to exclude
    df_aw = df_read_csv(od_anac_dir, file_od, list_col_exc, list_col_type_dic, None, csv_sep) # df_td -> dataframe contracting authorities
    df_print_details(df_aw, f"File '{file_od}'")
    print()
    # print(df_aw['flag_subappalto'].unique())
    df_aw['flag_subappalto'] = df_aw['flag_subappalto'].fillna(False)
    df_aw['flag_subappalto'] = df_aw['flag_subappalto'].astype(int)
    # print(df_aw['flag_subappalto'].unique())
    # print(df_aw['asta_elettronica'].unique())
    df_aw['asta_elettronica'] = df_aw['asta_elettronica'].replace({'0.0': 0, '1.0': 1})
    # Sostituire i NaN con 0
    df_aw['asta_elettronica'] = df_aw['asta_elettronica'].fillna(0)
    # Convertire i valori a interi
    df_aw['asta_elettronica'] = df_aw['asta_elettronica'].astype(int)
    # print(df_aw['asta_elettronica'].unique())    

    # EO
    file_od = "ECONOMIC_OPERATOR.csv"
    print("File:", file_od)
    file_path = Path(file_od)
    file_stem = file_path.stem # get the name without extension (is also the event name)

    # Read the file (dataset)
    list_col_exc = [] # no columns to exclude
    df_eo = df_read_csv(od_anac_dir, file_od, list_col_exc, list_col_type_dic, None, csv_sep) # df_td -> dataframe contracting authorities
    df_print_details(df_eo, f"File '{file_od}'")
    print()

    # VARIANTS
    file_od = "VARIANTS.csv"
    print("File:", file_od)
    file_path = Path(file_od)
    file_stem = file_path.stem # get the name without extension (is also the event name)

    # Read the file (dataset)
    list_col_exc = ["cod_motivo_variante","motivo_variante","data_approvazione_variante","id_aggiudicazione"] 
    df_va = df_read_csv(od_anac_dir, file_od, list_col_exc, list_col_type_dic, None, csv_sep) # df_va -> dataframe contracting authorities
    df_print_details(df_va, f"File '{file_od}'")
    print()
    # get all the cigs to use them
    list_cig_var = sorted(df_va["cig"].unique().tolist())

    # TENDERS
    file_od = "TENDER_NOTICE.csv"
    print("File:", file_od)
    file_path = Path(file_od)
    file_stem = file_path.stem # get the name without extension (is also the event name)

    # Get the columns to be included in stats by file name
    list_col_stats_inc = get_values_from_dict_list(list_col_stats_dic, file_od)
    list_col_stats_inc_len = len(list_col_stats_inc)

    # Get the columns to be filtered by file name
    list_col_filters = get_values_from_dict_list(list_col_filters_dic, file_od)
    list_col_filters_len = len(list_col_filters)

    # Read the file (dataset)
    list_col_exc = [] # no columns to exclude
    df_td = df_read_csv(od_anac_dir, file_od, list_col_exc, list_col_type_dic, None, csv_sep) # df_td -> dataframe tanders
    df_print_details(df_td, f"File '{file_od}'")
    print()

    # For the main file tender_notice:
    # add the column "cpv_division" that takes the first two characters of "cod_cpv" if it's not null
    # add the column "accordo_quadro" (1 or 0)
    # from the column "settore" remove redundant "SETTORI "string
    if file_od == tender_main_file:
        print(f"> Updating main tender file '{file_od}'")
        df_td['cpv_division'] = df_td['cod_cpv'].apply(lambda x: x[:2] if pd.notnull(x) else None)
        df_td['accordo_quadro'] = df_td['cig_accordo_quadro'].apply(lambda x: "1" if pd.notna(x) else "0")
        df_td['accordo_quadro'] = df_td['accordo_quadro'].astype('object')
        df_td['settore'] = df_td['settore'].str.replace('SETTORI ', '')
        df_td['sezione_regionale'] = df_td['sezione_regionale'].str.replace('SEZIONE REGIONALE  ', '')
        df_td['sezione_regionale'] = df_td['sezione_regionale'].str.replace('SEZIONE REGIONALE ', '')
        df_td['sezione_regionale'] = df_td['sezione_regionale'].str.replace('PROVINCIA AUTONOMA DI', 'PA')
        df_td['oggetto_principale_contratto'] = df_td['oggetto_principale_contratto'].str.replace('FORNITURE', 'U') # sUpplies
        df_td['oggetto_principale_contratto'] = df_td['oggetto_principale_contratto'].str.replace('SERVIZI', 'S') # Services
        df_td['oggetto_principale_contratto'] = df_td['oggetto_principale_contratto'].str.replace('LAVORI', 'W') # Work
        # Amount
        df_td['importo_lotto'] = pd.to_numeric(df_td['importo_lotto'], errors='coerce')
        # Rimuovere le righe con NaN (opzionale, puoi anche sostituirli con un valore predefinito)
        df_td = df_td.dropna(subset=['importo_lotto'])
        # Convertire la colonna in tipo float
        df_td['importo_lotto'] = df_td['importo_lotto'].astype(float)

        df_print_details(df_td, f"File '{file_od}' (after cleaning)")

    # Filters
    if file_od == tender_main_file and list_col_filters_len > 0:
        print(">> Applying filters")
        print(f"Filters applied ({list_col_filters_len}):", list_col_filters)
        
        # # Initial filter to preserve rows with 'cig' in list_cig_var
        preserve_cig_rows = df_td[df_td['cig'].isin(list_cig_var)]

        for filter_dict in list_col_filters:
            for key, value in filter_dict.items():
                # print("Distinct values before filtering:", list(df_od[key].unique())) # debug
                print("Filter key:", key, "filter value:", value) # debug
                df_td = df_td[df_td[key].isin(value)]
                # print("DF size after filter:", df_od.shape) # debug
        
        # Merge initial filter for 'cig' with applied filters to keep "variants"
        df_td = pd.concat([df_td, preserve_cig_rows]).drop_duplicates().reset_index(drop=True)

        df_print_details(df_td, f"File '{file_od}' (after filtering)")
        print("Distinct values filtered")
        for filter_dict in list_col_filters:
            for key, value in filter_dict.items():
                print("Column:", key)
                print(df_td[key].unique())
        print()

    # Drop columns
    col_drop_list = ['stato', 'id_centro_costo', 'denominazione_centro_costo', 'mese_pubblicazione', 'flag_prevalente', 'oggetto_gara']
    df_td = df_td.drop(columns=col_drop_list)
    
    # Regions fix
    print(">> Region fix")
    # df_td['sezione_regionale'] = df_td['sezione_regionale'].str.capitalize()
    df_td['sezione_regionale'] = df_td['sezione_regionale'].apply(capitalize_region)
    df_td = df_td[df_td['sezione_regionale'] != 'Non classificato'] # drop "Non classificato"
    print(df_td['sezione_regionale'].unique())
    print()

    # Join with CA x ISTAT
    print(">> Join Tenders with CA x ISTAT")
    df_td_final = pd.merge(left=df_td, right=df_ca_istat, left_on="cf_amministrazione_appaltante", right_on="codice_fiscale", how="left")
    df_td_final = clean_dataframe(df_td_final)
    df_print_details(df_td_final, f"File 'TENDER x CA x ISTAT'")
    print()

    # Join with AWARD
    print(">> Join Tenders with AWARD")
    df_td_final = pd.merge(left=df_td_final, right=df_aw, left_on="cig", right_on="cig", how="left")
    df_td_final = clean_dataframe(df_td_final)
    df_print_details(df_td_final, f"File 'TENDER x CA x ISTAT x AWARD'")
    print()

    # Join with EO
    print(">> Join Tenders with EO")
    df_td_final = pd.merge(left=df_td_final, right=df_eo, left_on="cig", right_on="cig", how="left")
    df_td_final = clean_dataframe(df_td_final)
    df_print_details(df_td_final, f"File 'TENDER x CA x ISTAT x AWARD'")
    print()

    # Join with VARIANTS
    print(">> Join Tenders with VARIANTS")
    df_td_final = pd.merge(left=df_td_final, right=df_va, left_on="cig", right_on="cig", how="left")
    df_td_final = clean_dataframe(df_td_final)
    # Set 'id_variante' to '1' if not null, otherwise to '0'
    df_td_final['id_variante'] = df_td_final['id_variante'].apply(lambda x: '1' if pd.notnull(x) else '0')
    df_print_details(df_td_final, f"File 'TENDER x CA x ISTAT x AWARD x VARIANTS'")
    print()

    # Save tender df with oggetto_lotto
    file_out = f"{file_stem}_filtered_with_objects.csv"
    path_out = Path(od_anac_dir) / file_out
    print("Saving filtered and joint dataframe (with oggetto_lotto) to:", path_out)
    df_td_final.to_csv(path_out, index=False, sep=";", quoting=csv.QUOTE_ALL)
    print()

    # Save tender df without oggetto_lotto
    col_drop_list = ['oggetto_lotto']
    df_td = df_td.drop(columns=col_drop_list)
    file_out = f"{file_stem}_filtered.csv"
    path_out = Path(od_anac_dir) / file_out
    print("Saving filtered and joint dataframe (without oggetto_lotto) to:", path_out)
    df_td_final.to_csv(path_out, index=False, sep=";", quoting=csv.QUOTE_ALL)
    print()

    print(">> Creating maps")
    plot_italy_region_map(df_td_final, "sezione_regionale", "importo_lotto", plot_dir)
    print()

    if stats_do == 1:
        # Stats 1 - Missing values
        print(">> Creating stats")
        print("> Missing values")
        dic_od = summarize_dataframe_to_dict(df_td_final, file_od)
        # print(dic_od) # debug
        df_stats = summarize_dataframe_to_df(dic_od)
        # print(df_stats.head()) # debug
        print("> Saving stats")
        save_stats(df_stats, file_stem, "_stats_missing", stats_dir)
        print()

        # Stats 2 - Distinct values
        print("> Distinct values")
        print("Colums included for this stat:", list_col_stats_inc_len)
        print(list_col_stats_inc) # debug
        if list_col_stats_inc_len > 0:
            df_stats = distinct_values_frequencies(df_td_final, list_col_stats_inc)
            # print(df_stats.head()) # debug
            print("> Saving stats")
            save_stats(df_stats, file_stem, "_stats_distinct", stats_dir)
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