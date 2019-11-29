import os
import numpy as np
import matplotlib.pyplot as plt


experiment_root_directory = os.path.join(
    './CIFAR-10',
    '201911290936',
)

subset_sizes = [0.1, 0.5, 1]

fig, ax = plt.subplots(len(subset_sizes), 1)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt_subplot_current_row = 0

for subset in subset_sizes:

    plt_subplot_current_column = 0

    print(str((plt_subplot_current_row, plt_subplot_current_column)))
    print(str(subset))

    # Open the score file
    score_file_path = os.path.join(
        experiment_root_directory,
        str(subset) + '_' + '0.5' + '_' + '0.0005',
        'accuracy_mc.txt',
    )

    scores = np.loadtxt(score_file_path).T
    
    print(scores)
    
    ax[plt_subplot_current_row].scatter(scores[0], scores[1])
    
    ax[plt_subplot_current_row].set_title(str(subset))

    plt_subplot_current_row += 1

plt.show()
