import pickle
import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

target_dir = os.getcwd()+'/data/'

def df_to_txt(input_df, targetdir, filename):
    """ Save the training and test set as .txt to make a training/valid/test set """
    save_as = os.path.join(targetdir, filename)
    return pd.DataFrame(input_df).to_csv(save_as,index=False, encoding='utf-8', header=False)

def split_data(input_df):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # print(os.path)
    datapath = os.getcwd()+'/raw_data/'

    with open(datapath+input_df, 'r') as file:
        lines = file.readlines()  

    if input_df == 'cleaned_parallel_sentences.txt':
        train, test = train_test_split(lines, test_size=0.3, random_state=42)
        test, valid = train_test_split(test, test_size=0.5, random_state=42)
        print(f"The data is split into {len(train)} of training sentences, {len(valid)} of validation sentences, and {len(test)} of test sentences.")
        
        # save as text files
        save_to_dir = os.path.join(target_dir)

        df_to_txt(train, save_to_dir, "train_"+input_df)
        df_to_txt(test, save_to_dir, "test_"+input_df)
        df_to_txt(valid, save_to_dir, "valid_"+input_df)

    elif input_df == 'tatoeba_sentences.csv':
        test, valid = train_test_split(lines, test_size=0.5, random_state=42)
        print(f"The data is split into {len(test)} of testing sentences and {len(valid)} of validation sentences.")

        save_to_dir = os.path.join(target_dir)

        df_to_txt(test, save_to_dir, "test_"+input_df)
        df_to_txt(valid, save_to_dir, "valid_"+input_df)

def main():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--filename", type=str, help='Path to the input file (txt or csv)')
    args = parser.parse_args()

    split_data(args.filename)
    
if __name__ == '__main__':
    main()
    