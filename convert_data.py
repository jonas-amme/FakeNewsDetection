
import argparse
import torch
from LoadData import LoadData



# initialize arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/data/s1805819/fakenewsnet_dataset/cascades', help='enter your data path')
parser.add_argument('--model_path', type=str, default='/data/s2583550/FakeNewsDetection/model/', help='enter your model path')
args = parser.parse_args()


if __name__ == '__main__':

    # Load Data
    dataloader = LoadData(args.data_path)  # args.data_path = data path of twitter data, to be specified in beginning
    graph_data = dataloader.graph_data

    # save data
    torch.save(args.model_path + 'graph_data.pt')
