from train_result.config import task_folder, batch_folder, computer
import os

result_path = os.path.join("../result", "{0}-{1}/{0}-{1}_result.csv".format(task_folder, batch_folder))
if not os.path.exists(result_path):
    result_path = os.path.join("../result", "{}-{}_result_{}.csv".format(task_folder, batch_folder, computer))

trained_models = []
with open(result_path, "r") as f:
    for line in f.readlines():
        try:
            trained_models.append(int(line.split(",")[0]))
        except:
            continue
# print(sorted(trained_models))

plan_path = os.path.join(task_folder, batch_folder, "{}.csv".format(batch_folder))
plan_models = []
with open(plan_path, "r") as f:
    for idx, line in enumerate(f.readlines()):
        if len(line) > 30 and "backbone" not in line:
            plan_models.append(idx)
# print(plan_models)

remain_models = [model for model in plan_models if model not in trained_models]
print(remain_models)
wrong_models = [model for model in trained_models if model not in plan_models]
print(wrong_models)

# trained_folder = "{}/result/log_result".format(models_name)
#
# import os
# file_trained = [int(file) for file in os.listdir(trained_folder)]
#
# # print(sorted(trained))
# rest_file = [item for item in models if item not in file_trained]
# print("Not in files:")
# print(rest_file)
#
# train_log_name = "alphapose_aic/result/aic_origin_result.xlsx"
# wb = openpyxl.load_workbook(train_log_name)
# sheet_names = wb.get_sheet_names()
# ws = wb.get_sheet_by_name(sheet_names[0])
# logs = []
# for row in range(ws.max_row-1):
#     if ws.cell(row+2,1).value is not None:
#         logs.append(ws.cell(row+2,1).value)
# wb.close()
# print("Not in logs:")
# rest_log = [item for item in models if item not in logs]
# print(rest_log)
#
# rest_all = [item for item in rest_file if item in rest_log]
# print("Not in both:")
# print(rest_all)
