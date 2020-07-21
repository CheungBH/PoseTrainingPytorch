
model_txt = "training_csv/alphapose_aic.txt"

models = []
with open(model_txt, "r") as f:
    for line in f.readlines():
        try:
            models.append(int(line.split(" ")[-1][:-2]))
        except:
            continue

print(len(models))

