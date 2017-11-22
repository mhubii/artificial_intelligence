import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Build the data frame
df = pd.DataFrame({'from':['Uta', 'Sabine', 'Biagio', 'Timo', 'Plymouth Driver', 'BMW Driver', 'Tuna Eater', 'Shrimp Eater',
                           'Dog Owner', 'Fish Owner', 'Dog Owner', 'BMW Driver', 'Peugeot Driver', 'Gyoza', 'Fanta'],
                   'to':['Kia', 'Monkey', 'Cola', 'Most right sit', 'Most left sit', 'Beer Drinker', 'Salmon Eater', 'Ilama',
                        'Tuna Eater', 'Fanta', 'Gyoza', 'Sprite', 'Shrimp Eater', 'Beer Drinker', 'Ilama']})

# Edges
#edge_attributes = ['drives', 'has a', 'drinks', 'sits on', 'sits to the left of', 'sit next to each other',
#                   'has a neighbour with a pet', 'are not neighbours', 'has a neighbour who drinks',
#                   'eats', 'drinks', 'sits to the right of', 'drinks', 'keeps a pet']

# Build the graph
G = nx.from_pandas_dataframe(df, 'from', 'to')

# Graph with Custom nodes:
nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=40)
plt.show()
