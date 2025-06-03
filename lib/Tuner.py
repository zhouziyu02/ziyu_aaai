import torch
from collections import defaultdict
import pandas as pd
import optuna
import datetime
import os

# 根据你的项目实际情况导入train_main
from tPatchGNN.exp import train_main  # 如果路径不同，请更改

class Tuner:
    """Tuner for tPatchGNN using Optuna."""
    def __init__(self, ranSeed, n_jobs):
        self.fixedSeed = ranSeed
        self.n_jobs = n_jobs
        self.result_dic = defaultdict(list)
        self.current_time = datetime.datetime.now()
        # 处理时间戳字符串，避免文件名有冒号
        self.current_time = str(self.current_time).replace(':', '-').split('.')[0]

    def optuna_objective(self, trial, args):
        # 1) 定义要搜索的超参数空间
        args.batch_size = trial.suggest_categorical('batch_size', [32,64,128,256])
        args.lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2])
        # args.patch_size = trial.suggest_categorical('patch_size', [300])
        args.hid_dim = trial.suggest_categorical('hid_dim', [16, 32, 64, 128])
        args.K = trial.suggest_categorical('K', [2, 4, 8])
        args.te_dim = trial.suggest_categorical('te_dim', [10, 20, 30])
        # args.random_dim = trial.suggest_categorical('random_dim', [32, 64])
        args.nlayer = trial.suggest_categorical('nlayer', [1, 2, 3])
        # args.seed = trial.suggest_int('seed', 0, 1, 3407)

        # 2) 仅用于显示，构建一个描述当前Trial设置的字符串
        setting = (
            f"{args.model}_{args.dataset}_bs{args.batch_size}"
            f"_nlayer{args.nlayer}_K{args.K}_te{args.te_dim}_hdim{args.hid_dim}"
            f"_lr{args.lr:.1e}"
        )
        print(f">>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")

        # 3) 调用训练函数，拿到最终test_mse
        final_test_mse = train_main(args, optunaTrialReport=trial)

        # 4) 返回数值给Optuna，作为目标函数要最小化的值
        return final_test_mse

    def tune(self, args):
        """
        执行Optuna的超参数搜索流程
        """
        # 设定最大搜索次数
        n = 120

        # 如果已有 self.study，先删除，防止重复使用
        try:
            del self.study
            print('Deleted previous tuner object.')
        except AttributeError:
            print('No previous tuner object found; will create a new one.')

        # 创建study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.fixedSeed)
        )

        wrapped_objective = lambda trial: self.optuna_objective(trial, args)
        self.study.optimize(wrapped_objective, n_trials=n, n_jobs=self.n_jobs)

        self.save_result(args)

    def save_result(self, args):
        """
        将最优结果与参数写入CSV文件
        """
        os.makedirs('./hyperParameterSearchOutput/', exist_ok=True)

        file_name = f"{args.model}_{args.dataset}"
        data = args.dataset


        best_trial = self.study.best_trial
        best_params = self.study.best_params
        best_result = best_trial.value  # 即最优 trial 的目标函数值

        # 存入字典
        self.result_dic['data'].append(data)

        self.result_dic['loss'].append(best_result)

        for key, value in best_params.items():
            self.result_dic[key].append(value)

        result_df = pd.DataFrame(self.result_dic)
        out_csv_path = os.path.join(
            './hyperParameterSearchOutput/',
            file_name + f'_bst_parms_{self.current_time}.csv'
        )

        try:
            result_df.to_csv(out_csv_path, index=False)
            print(f"Best params saved to: {out_csv_path}")
        except Exception as e:
            print('[Error] Could not save best params CSV (is the file open?)')
            print(e)

        print(result_df)
