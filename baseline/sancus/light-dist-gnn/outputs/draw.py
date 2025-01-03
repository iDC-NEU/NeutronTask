import sys
import os
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python plot_accuracy.py <filename> <output_folder>")
    sys.exit(1)

filename = sys.argv[1]
output_folder = sys.argv[2]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(filename, "r") as file:
    accuracy_data = file.read().strip().split()

accuracy_data = [float(accuracy) for accuracy in accuracy_data]

epochs = list(range(1, len(accuracy_data) + 1))

# 绘制图像
plt.plot(epochs, accuracy_data, marker='o')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

output_filename = os.path.join(output_folder, "accuracy_plot.svg")
plt.savefig(output_filename, format='svg')
plt.show()

# python draw.py cora/output_cora2.txt cora