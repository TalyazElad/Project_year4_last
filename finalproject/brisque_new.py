from brisque import BRISQUE
from PIL import Image
import numpy as np

# Define the directory where the images are stored

# Initialize the BRISQUE object
brisq = BRISQUE()

# Define a function to compute the score for a given image
def compute_score(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    score = brisq.score(img)
    if score == np.nan:
        score = 100
        #print("Brisque returned nan")
    #print(f"{score:.2f}")
    return("{}".format(score))

# Compute the scores in parallel using multithreading
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     scores = list(executor.map(compute_score, [f"frame{i}.jpg" for i in range(1, 849)]))
#
# # Calculate the average score using the mean function from the statistics module
# average_score = statistics.mean(scores)
# print(f"Average BRISQUE score: {average_score:.2f}")

