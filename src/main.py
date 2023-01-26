import pandas as pd
import pickle


if __name__ == '__main__':
    
    df = pd.read_csv('./data/matrix/0 file.csv')

    nodes = df.columns.to_list()    

    ady_list = [[] for _ in range(len(nodes))]

    for i in range(df.shape[0]):
        row = df.loc[i].to_list()
        for j in range(len(row)):
            if row[j] != 0:
                ady_list[i].append((j, row[j]))


    with open('./data/adym_0.pkl', 'wb') as f:
        pickle.dump(ady_list, f)

    a = pickle.load(open('./data/adym_0.pkl', 'rb'))

    print(a)

    
  