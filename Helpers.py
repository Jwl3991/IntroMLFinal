import numpy as np
import pandas as pd
from numpy import savetxt
import csv
from datetime import datetime


def normalize_volume(one_csv='no'):

    if one_csv == 'no':
        # Read CSV file
        csv_files = ["TSLA_HP.csv", 'AAPL_HP.csv', "GOOG_HP.csv", "BRKA_HP.csv", "DOW_HP.csv", "ENPH_HP.csv",
                  "EURO_HP.csv", "GME_HP.csv", "GOLD_HP.csv", "MSFT_HP.csv", "NASDAQ_HP.csv",
                  "SP500_HP.csv", "USO_HP.csv"]

        for file_name in csv_files:
            volume_type = 0

            df = pd.read_csv(file_name)

            # Extract 'Volume' column
            try:
                volume = df['Volume']
                volume_type = 0
            except:
                if 'Volume' in df.columns or ' Volume' in df.columns:
                    volume = df[' Volume']
                    volume_type = 1
                else:
                    continue

            # Normalize the 'Volume' column
            volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())

            # Replace the original 'Volume' column with the normalized one
            if volume_type == 0:
                df['Volume'] = volume_normalized
            else:
                df[' Volume'] = volume_normalized

            # Save the new DataFrame with normalized 'Volume' column to a new CSV file
            output_file_name = file_name
            df.to_csv(output_file_name, index=False)
    else:
        df = pd.read_csv(one_csv)
        volume_type = 0
        # Extract 'Volume' column
        try:
            volume = df['Volume']
            volume_type = 0
        except:
            if 'Volume' in df.columns or ' Volume' in df.columns:
                volume = df[' Volume']
                volume_type = 1
            else:
                print("cannot find volume column in", one_csv)

        # Normalize the 'Volume' column
        volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())

        # Replace the original 'Volume' column with the normalized one
        if volume_type == 0:
            df['Volume'] = volume_normalized
        else:
            df[' Volume'] = volume_normalized

        # Save the new DataFrame with normalized 'Volume' column to a new CSV file
        output_file_name = one_csv + "_VN"
        df.to_csv(output_file_name, index=False)

# List of CSV file names


# Iterate over the list of CSV files and normalize the 'Volume' column
def prepend_20_to_year(date_string):
    if len(date_string.split('/')[-1]) == 2:
        return '/'.join(date_string.split('/')[:-1]) + '/20' + date_string.split('/')[-1]
    else:
        return date_string


def removeBadDates():
    # Read the first CSV file and store the dates from the first column
    first_file = "TSLA_HP.csv"
    dates_data = pd.read_csv(first_file)
    stored_dates = dates_data['Date'].apply(lambda x: datetime.strptime(x.strip(), '%m/%d/%Y')).tolist()

    # List of other CSV file names
    other_files = ["AAPL_HP.csv", "GOOG_HP.csv", "BRKA_HP.csv", "DOW_HP.csv", "ENPH_HP.csv",
                  "EURO_HP.csv", "GME_HP.csv", "GOLD_HP.csv", "MSFT_HP.csv", "NASDAQ_HP.csv",
                  "SP500_HP.csv", "USO_HP.csv"]   # Add more file names as needed

    for file_name in other_files:
        # Read the CSV file
        data = pd.read_csv(file_name)
        data['Date'] = data['Date'].apply(prepend_20_to_year)
        data = data[data['Date'].apply(lambda x: datetime.strptime(x.strip(), '%m/%d/%Y') in stored_dates)]
        # Remove rows where the date is not in the stored dates
        #data = data[data.iloc[:, 0].isin(stored_dates)]


        # Save the updated CSV file
        data.to_csv(file_name, index=False)


def consolidateData():
    # List of file names
    file_names = ["TSLA_HP.csv", 'AAPL_HP.csv', "GOOG_HP.csv", "BRKA_HP.csv", "DOW_HP.csv", "ENPH_HP.csv",
                  "EURO_HP.csv", "GME_HP.csv", "GOLD_HP.csv", "MSFT_HP.csv", "NASDAQ_HP.csv",
                  "SP500_HP.csv", "USO_HP.csv"]  # Add more file names as needed

    consolidated_data = pd.DataFrame()

    for file_name in file_names:
        # Read the CSV file
        data = pd.read_csv(file_name)

        # Extract the stock symbol from the file name (e.g., 'TSLA' from 'TSLA_HP.csv')
        stock_symbol = file_name.split("_")[0]

        # Rename the columns
        data.columns = ['date'] + [f'{stock_symbol} {col.capitalize()}' for col in data.columns[1:]]

        # Concatenate the data horizontally into the consolidated DataFrame
        if consolidated_data.empty:
            consolidated_data = data
        else:
            data = data.drop(columns=['date'])  # Remove the 'date' column from all but the first file
            consolidated_data = pd.concat([consolidated_data, data], axis=1)

    # Save the consolidated data to a new CSV file
    consolidated_data.to_csv("consolidated_data.csv", index=False)
def setDates():
    csv_file = "consolidated_data.csv"
    #csv_file = "AAPL_HP_TEST.csv"
    data = pd.read_csv(csv_file)

    # Parse the date column
    data['date'] = pd.to_datetime(data['date'])

    # Extract day, month, and year from the date column
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # Drop the original date column
    data = data.drop('date', axis=1)

    # Reorder columns to have day, month, and year as the first three columns
    cols = ['day', 'month', 'year'] + [col for col in data.columns if col not in ['day', 'month', 'year']]
    data = data[cols]

    # Save the updated DataFrame to a new CSV file
    data.to_csv("updated_file.csv", index=False)
    #data.to_csv("AAPL_HP_TEST.csv", index=False)

def reformat():
    file_names = ["TSLA_HP.csv", 'AAPL_HP.csv', "GOOG_HP.csv", "BRKA_HP.csv", "DOW_HP.csv", "ENPH_HP.csv",
                  "EURO_HP.csv", "GME_HP.csv", "GOLD_HP.csv", "MSFT_HP.csv", "NASDAQ_HP.csv",
                  "SP500_HP.csv", "USO_HP.csv"]
    csv_file = "consolidated_data.csv"
    data = pd.read_csv("updated_file.csv")
    orig_data = data.to_numpy()
    new_data = data.columns.to_numpy()
    new_data = np.insert(new_data, 0, "Target")
    new_data = np.insert(new_data, 0, "Stock")
    new_data = np.append(new_data, "Y")
    new_data = new_data.reshape((1,-1))
    offset = 0
    for i in range(len(file_names)):
        file = file_names[i]
        if i != 0:
            dataTMP = pd.read_csv(file)
            offset = len(dataTMP.columns) + offset -1

        for j in range(1, len(orig_data)):
            #tmp = np.array([])
            #tmp = np.append(tmp, i)
            for k in range(4):
                tmp = np.array([])
                tmp = np.append(tmp, i)
                tmp = np.append(tmp, k)
                tmp = np.append(tmp, orig_data[j])
                tmp = np.append(tmp, orig_data[j-1][3 + offset + k])
                new_data = np.concatenate((new_data, tmp.reshape(1, -1)), axis=0)

    saveData = np.ndarray.tolist(new_data)
    with open('data_reformatted.csv', 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)


        # Write the data rows
        csv_writer.writerows(saveData)

def removeBadCells():
    # Read the CSV file
    file_name = 'data_reformatted.csv'
    df = pd.read_csv(file_name)

    # Replace empty or NaN entries with 0
    df.fillna(0, inplace=True)

    # Save the modified DataFrame to a new CSV file
    output_file_name = file_name.split('.')[0] + '_modified.csv'
    df.to_csv(output_file_name, index=False)

#removeBadDates()
#normalize_volume()
#consolidateData()
#setDates()
#reformat()
#removeBadCells()