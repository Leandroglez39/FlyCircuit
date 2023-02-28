import pandas as pd
import pickle
import multiprocessing
import time

def temp():
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

def cpu_bound(number):
    return sum(i * i for i in range(number))


def find_sums(numbers):
    with multiprocessing.Pool(8) as pool:
        pool.map(cpu_bound, numbers)

def mdouble(x):
    return x * 2

if __name__ == '__main__':
    

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        A = pool.map(mdouble, [1, 2, 3])

    print(A)
    print(type(A))
    # numbers = [5000000 + x for x in range(20)]

    # start_time = time.time()
    # find_sums(numbers)
    # duration = time.time() - start_time
    # print(f"Duration {duration} seconds")

    
    
  