# Task
Classify 0/1 whether the image is flipped horizontally or not
Image is a matrix of any shape and 3 channels

1. Searching any task specific datasets is a dead thing - well, I could do that, but instead we can focus on some more fruitful approaches


# Deep learning

1. Create labels from CLIP - this would be the ground truth
2. Learn some small classifier on top of its labels

or

Just use some backbone, and train linear probe or SVM on top of it

# Unsupervised
1. Extract stuff via backbone
2. Reduce dimensionality
3. Classify