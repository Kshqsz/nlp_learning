from transformers import pipeline

clf = pipeline("sentiment-analysis")

result = clf("this movie is very boring!")

print(result)