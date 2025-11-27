from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model = "uer/roberta-base-finetuned-jd-binary-chinese"
)

texts = [
    "这个电影真的太好看了，我特别喜欢！",
    "这个手机太卡了，我非常失望。",
    "服务态度一般般。",
    "她让我滚蛋。"
]

for t in texts:
    result = clf(t)
    print(t, "=>", result)

