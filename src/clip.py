from transformers import pipeline

# load pipeline
ckpt = "google/siglip2-base-patch16-224"
image_classifier = pipeline(model=ckpt, task="zero-shot-image-classification")

# load image and candidate labels
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
candidate_labels = ["2 cats", "a plane", "a remote"]

# run inference
outputs = image_classifier(image, candidate_labels)
print(outputs)
