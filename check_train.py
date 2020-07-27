
model_txt = "training_csv/alphapose_aic.txt"

models = []
with open(model_txt, "r") as f:
    for line in f.readlines():
        try:
            models.append(int(line.split(" ")[-1][:-2]))
        except:
            continue

print(len(models))

trained_txt = "result/aic_origin_result.txt"

with open(trained_txt, "r") as t:
    trained = []
    for line in t.readlines()[1:]:
        trained.append(int((line.split(",")[-6]).split("/")[-1]))

print(sorted(trained))
rest = [item for item in models if item not in trained]
print(rest)
