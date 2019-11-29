import os
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

experiment_root_directory = os.path.join(
    './protein-tertiary-structure',
    '201911080145',
)

hidden_dims = [25, 50, 100]
n_layers = [1, 3, 5]

plt_subplot_current_row = 0

for hidden_dim in hidden_dims:

    plt_subplot_current_column = 0

    for n_layer in n_layers:
        print(str((plt_subplot_current_row, plt_subplot_current_column)))
        print(str((hidden_dim, n_layer)))

        # Open the score file
        score_file_path = os.path.join(
            experiment_root_directory,
            '0.6' + '_' + str(hidden_dim) + '_' + str(n_layer) + '_' + '0.1' + '_' + '0.05',
            'lls_mc.txt',
        )
    
        scores = np.loadtxt(score_file_path).T
        
        ax[plt_subplot_current_row, plt_subplot_current_column].scatter(scores[0], scores[1])
        
        ax[plt_subplot_current_row, plt_subplot_current_column].set_title(str((hidden_dim, n_layer)))
        

        

        plt_subplot_current_column += 1

    plt_subplot_current_row += 1
