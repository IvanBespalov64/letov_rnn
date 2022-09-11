from train import GenModel
from train import PoemsSet
from train import RNN_model

import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model")
    parser.add_argument("--prefix")
    parser.add_argument("--length")    
    
    args = parser.parse_args()
 
    model = None
    try:
        with open(args.model, "rb") as loaded_file:
            model = pickle.load(loaded_file)
    except FileNotFoundError:
        print("Wrong file name!")
        exit(0)

    print(model.generate(int(args.length), args.prefix))
