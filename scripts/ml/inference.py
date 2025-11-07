import pandas as pd
import numpy as np
import joblib
import sys
import os
import io


cols = ["version","tos","length","id","flags","ttl","protocol","header","source","destination","options", "vmid"]

def show_help():
    print("""Usage: predict -[h|f|s]

Predicts which os sends a packet

Available options:
-h, --help      Print this help and exit
-f, --file      Reads the input from a file
-s, --string    Reads the input from a string
""")
    sys.exit()

def prepare_data(raw_dataframe):   
    X_new = raw_dataframe.drop(columns=["source", "destination", "options", "header"])
    cols_to_convert = ['tos', 'length', 'id', 'ttl']
    for col in cols_to_convert:
        try:
            X_new[col] = X_new[col].astype(str).apply(lambda x: int(x, 16))
        except ValueError:
            try:
                X_new[col] = X_new[col].astype(str).apply(lambda x: int(x))
            except ValueError:
                pass        
    return X_new

def main():
    flag = sys.argv[1]
    input_type = None

    match flag:
        case "-f" | "--file":
            input_arg = sys.argv[2]
            print(f"Data from file: {input_arg}")
            raw_data = pd.read_csv(input_arg, header=None, names=cols)
            input_type = "file"

        case "-s" | "--string":
            input_arg = sys.argv[2]
            print(f"Data from string input: {input_arg}")
            string_data = io.StringIO(input_arg)
            raw_data = pd.read_csv(string_data, header=None, names=cols)
            input_type = "line"

        case "-h" | "--help":
            show_help()

        case _:
            print("Unknown flag. Try -h or --help for help.")
            sys.exit(1)

    print("Loading classification model...")
    model = joblib.load('models/best_model_RandomForest.joblib') 

    print(f"Loaded {raw_data.shape[0]} packet(s).")

    X_processed = prepare_data(raw_data)

    predictions = model.predict(X_processed)

    if input_type == 'file':
        print(list(predictions))
        results_df = raw_data.copy()
        results_df['predicted_vmid'] = predictions
        results_df.to_csv("classified_results.csv", index=False)
        print("\nresults saved to classified_results.csv")
            
    else: 
        print(predictions[0])

if __name__ == "__main__":
    main()
