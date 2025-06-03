import os
import matplotlib.pyplot as plt
import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device

# Adapted from: https://github.com/rtqichen/txime-series-datasets

class PhysioNet(object):

	urls = [
		'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
		'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
		'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download',
	]

	params = [
		'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
		'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
		'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
		'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
	]

	params_dict = {k: i for i, k in enumerate(params)}

	labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
	labels_dict = {k: i for i, k in enumerate(labels)}

	def __init__(self, root, download = False,
		quantization = None, n_samples = None, device = torch.device("cpu")):

		self.root = root
		self.reduce = "average"
		self.quantization = quantization

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if device == torch.device("cpu"):
			data_a = torch.load(os.path.join(self.processed_folder, self.set_a), map_location='cpu')
			data_b = torch.load(os.path.join(self.processed_folder, self.set_b), map_location='cpu')
			data_c = torch.load(os.path.join(self.processed_folder, self.set_c), map_location='cpu')
		else:
			data_a = torch.load(os.path.join(self.processed_folder, self.set_a))
			data_b = torch.load(os.path.join(self.processed_folder, self.set_b))
			data_c = torch.load(os.path.join(self.processed_folder, self.set_c))

		self.data = data_a + data_b + data_c # a list with length 12000

		if n_samples is not None:
			print('Total records:', len(self.data))
			self.data = self.data[:n_samples]

	def download(self):
		if self._check_exists():
			return

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)

		for url in self.urls:
			filename = url.rpartition('/')[2]
			download_url(url, self.raw_folder, filename, None)
			tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
			tar.extractall(self.raw_folder)
			tar.close()

			print('Processing {}...'.format(filename))

			dirname = os.path.join(self.raw_folder, filename.split('.')[0])
			patients = []
			total = 0
			for txtfile in os.listdir(dirname):
				record_id = txtfile.split('.')[0]
				with open(os.path.join(dirname, txtfile)) as f:
					lines = f.readlines()
					prev_time = 0
					tt = [0.]
					vals = [torch.zeros(len(self.params))]
					mask = [torch.zeros(len(self.params))]
					nobs = [torch.zeros(len(self.params))]
					for l in lines[1:]:
						total += 1
						time, param, val = l.split(',')
						# Time in hours
						time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.

						# round up the time stamps (up to 6 min by default)
						# used for speed -- we actually don't need to quantize it in Latent ODE
						if(self.quantization != None and self.quantization != 0):
							time = round(time / self.quantization) * self.quantization

						if time != prev_time:
							tt.append(time)
							vals.append(torch.zeros(len(self.params)))
							mask.append(torch.zeros(len(self.params)))
							nobs.append(torch.zeros(len(self.params)))
							prev_time = time

						if param in self.params_dict:
							n_observations = nobs[-1][self.params_dict[param]]
							if self.reduce == 'average' and n_observations > 0:
								prev_val = vals[-1][self.params_dict[param]]
								new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
								vals[-1][self.params_dict[param]] = new_val
							else:
								vals[-1][self.params_dict[param]] = float(val)
							mask[-1][self.params_dict[param]] = 1
							nobs[-1][self.params_dict[param]] += 1
						else:
							assert (param == 'RecordID' or param ==''), 'Read unexpected param {}'.format(param)

				tt = torch.tensor(tt).to(self.device)
				vals = torch.stack(vals).to(self.device)
				mask = torch.stack(mask).to(self.device)

				patients.append((record_id, tt, vals, mask))

			torch.save(
				patients,
				os.path.join(self.processed_folder, 
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			)
				
		print('Done!')

	def _check_exists(self):
		for url in self.urls:
			filename = url.rpartition('/')[2]

			if not os.path.exists(
				os.path.join(self.processed_folder, 
					filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
			):
				return False
		return True

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, 'processed')

	@property
	def set_a(self):
		return 'set-a_{}.pt'.format(self.quantization)

	@property
	def set_b(self):
		return 'set-b_{}.pt'.format(self.quantization)
	
	@property
	def set_c(self):
		return 'set-c_{}.pt'.format(self.quantization)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def get_label(self, record_id):
		return self.labels[record_id]

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Quantization: {}\n'.format(self.quantization)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str

	def visualize(self, timesteps, data, mask, plot_name):
		width = 15
		height = 15

		non_zero_attributes = (torch.sum(mask,0) > 2).numpy()
		non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
		n_non_zero = sum(non_zero_attributes)

		mask = mask[:, non_zero_idx]
		data = data[:, non_zero_idx]
		
		params_non_zero = [self.params[i] for i in non_zero_idx]
		params_dict = {k: i for i, k in enumerate(params_non_zero)}

		n_col = 3
		n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
		fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

		#for i in range(len(self.params)):
		for i in range(n_non_zero):
			param = params_non_zero[i]
			param_id = params_dict[param]

			tp_mask = mask[:,param_id].long()

			tp_cur_param = timesteps[tp_mask == 1.]
			data_cur_param = data[tp_mask == 1., param_id]

			ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(),  marker='o') 
			ax_list[i // n_col, i % n_col].set_title(param)

		fig.tight_layout()
		fig.savefig(plot_name)
		plt.close(fig)

def get_data_min_max(records, device):
	inf = torch.Tensor([float("Inf")])[0].to(device)

	data_min, data_max, time_max = None, None, -inf

	for b, (record_id, tt, vals, mask) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

		time_max = torch.max(time_max, tt.max())

	print('data_max:', data_max)
	print('data_min:', data_min)
	print('time_max:', time_max)

	return data_min, data_max, time_max

def get_seq_length(args, records):
	
	max_input_len = 0
	max_pred_len = 0
	lens = []
	for b, (record_id, tt, vals, mask) in enumerate(records):
		n_observed_tp = torch.lt(tt, args.history).sum()
		max_input_len = max(max_input_len, n_observed_tp)
		max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
		lens.append(n_observed_tp)
	lens = torch.stack(lens, dim=0)
	median_len = lens.median()

	return max_input_len, max_pred_len, median_len


def patch_variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
	Data form as input:
		batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
		batch_vals: (B, M, L_in, D) tensor containing the observed values.
		batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
	Data form to predict:
		flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
		flat_vals: (B, L_out, D) tensor containing the observed values.
		flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	# print("Transformable Patching")
	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)

	# the number of observed time points 
	n_observed_tp = torch.lt(combined_tt, args.history).sum()
	observed_tp = combined_tt[:n_observed_tp] # (n_observed_tp, )

	patch_indices = []
	st, ed = 0, args.patch_size
	for i in range(args.npatch):
		if(i == args.npatch-1):
			inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
		else:
			inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
		patch_indices.append(inds)
		st += args.stride
		ed += args.stride

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 
	for b, (record_id, tt, vals, mask) in enumerate(batch):
		indices = inverse_indices[offset:offset+len(tt)]
		offset += len(tt)
		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		tmp_n_observed_tp = torch.lt(tt, args.history).sum()
		predicted_tp.append(tt[tmp_n_observed_tp:])
		predicted_data.append(vals[tmp_n_observed_tp:])
		predicted_mask.append(mask[tmp_n_observed_tp:])

	combined_tt = combined_tt[:n_observed_tp]
	combined_vals = combined_vals[:, :n_observed_tp]
	combined_mask = combined_mask[:, :n_observed_tp]
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	if(args.dataset != 'ushcn'):
		combined_vals = utils.normalize_masked_data(combined_vals, combined_mask, 
			att_min = data_min, att_max = data_max)
		predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask, 
			att_min = data_min, att_max = data_max)

	combined_tt = utils.normalize_masked_tp(combined_tt, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
		
	data_dict = {
		"data": combined_vals, # (n_batch, T_o, D)
		"time_steps": combined_tt, # (T_o, )
		"mask": combined_mask, # (n_batch, T_o, D)
		"data_to_predict": predicted_data,
		"tp_to_predict": predicted_tp,
		"mask_predicted_data": predicted_mask,
		}

	data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)

	return data_dict
	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	
	return data_dict

def variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	 
	observed_tp = []
	observed_data = []
	observed_mask = [] 
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 

	for b, (record_id, tt, vals, mask) in enumerate(batch):
		n_observed_tp = torch.lt(tt, args.history).sum()
		observed_tp.append(tt[:n_observed_tp])
		observed_data.append(vals[:n_observed_tp])
		observed_mask.append(mask[:n_observed_tp])
		
		predicted_tp.append(tt[n_observed_tp:])
		predicted_data.append(vals[n_observed_tp:])
		predicted_mask.append(mask[n_observed_tp:])

	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	if(args.dataset != 'ushcn'):
		observed_data = utils.normalize_masked_data(observed_data, observed_mask, 
			att_min = data_min, att_max = data_max)
		predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask, 
			att_min = data_min, att_max = data_max)
	
	observed_tp = utils.normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
		
	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	
	return data_dict


 


# ===== 1. 依据时间间隔自适应推算观测阈值 =========================
def _calibrate_k(tt_in_window, base_window,
                 alpha=0.4, beta=1.6,
                 k_min_floor=2, k_gap=4):
    """
    根据窗口内的中位 inter‑arrival 推算 k_min / k_max
    """
    if len(tt_in_window) < 2:
        # 极端稀疏：使用最小阈值
        return k_min_floor, k_min_floor + k_gap

    dts = tt_in_window[1:] - tt_in_window[:-1]
    med_dt = torch.median(dts)
    # 预期该时间窗内能看到的点数
    n_ref = (base_window / med_dt).item()
    k_min = max(k_min_floor, int(n_ref * alpha + 0.5))
    k_max = max(k_min + k_gap, int(n_ref * beta  + 0.5))
    return k_min, k_max

# ===== 2. 自适应细 patch 拆分 =========================
def _adaptive_indices(tt, k_min, k_max, s_min, s_max):
    ptr, n = 0, len(tt)
    idx_list = []
    while ptr < n:
        w = s_min
        while True:                               # 向右扩张
            right = torch.searchsorted(tt, tt[ptr] + w)
            cnt = right - ptr
            if cnt >= k_min or w >= s_max:
                break
            w *= 2
        if cnt == 0:                              # 空窗保护
            ptr += 1; continue
        if cnt > k_max:                           # 过密 → 等分
            n_split = (cnt + k_max - 1) // k_max
            step = cnt // n_split
            for j in range(n_split):
                l = ptr + j * step
                r = ptr + cnt if j == n_split - 1 else l + step
                idx_list.append(torch.arange(l, r, device=tt.device))
            ptr += cnt
        else:
            idx_list.append(torch.arange(ptr, right, device=tt.device))
            ptr = right
    return idx_list

# ===== 3. 主 collate 函数 =========================
def adaptive_multiscale_collate_fn(batch, args, device=torch.device('cpu'),
                                   data_type="train",
                                   data_min=None, data_max=None, time_max=None):
    # print("Adaptive‑Multiscale Collate >>", data_type)
    D      = batch[0][2].shape[1]
    stride = args.stride                       # baseline stride / patch_size
    patch_size = args.patch_size
    s_min = stride * getattr(args, 's_min_rel', 0.5)
    s_max = stride * getattr(args, 's_max_rel', 4.0)
    g     = getattr(args, 'coarse_group', 4)   # coarse 聚合阶

    # --------------- 合并批次时间戳 ----------------
    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    n_obs_tp = torch.lt(combined_tt, args.history).sum()
    observed_tp_all = combined_tt[:n_obs_tp]                   # (T_o,)

    # --------------- auto‑calibrate k values -------
    k_min, k_max = _calibrate_k(observed_tp_all, patch_size)

    # --------------- 生成两级 patch ----------------
    lvl1 = _adaptive_indices(observed_tp_all, k_min, k_max, s_min, s_max)
    lvl2 = [torch.cat(lvl1[i:i+g]) for i in range(0, len(lvl1), g)]
    patch_indices = lvl1 + lvl2
    args.npatch   = len(patch_indices)        # 通知 utils.split_and_patch_batch

    # --------------- 将观测映射到统一时间轴 ----------
    B = len(batch)
    combined_vals = torch.zeros(B, len(combined_tt), D, device=device)
    combined_mask = torch.zeros_like(combined_vals)

    pred_tp, pred_val, pred_mask = [], [], []
    offset = 0
    for b_idx, (_, tt, vals, mask) in enumerate(batch):
        idx = inverse_indices[offset: offset + len(tt)]
        offset += len(tt)
        combined_vals[b_idx, idx]  = vals
        combined_mask[b_idx, idx]  = mask

        n_obs = torch.lt(tt, args.history).sum()
        pred_tp.append(tt[n_obs:])
        pred_val.append(vals[n_obs:])
        pred_mask.append(mask[n_obs:])

    combined_tt   = combined_tt[:n_obs_tp]
    combined_vals = combined_vals[:, :n_obs_tp]
    combined_mask = combined_mask[:, :n_obs_tp]

    pred_tp    = pad_sequence(pred_tp, batch_first=True)
    pred_val   = pad_sequence(pred_val, batch_first=True)
    pred_mask  = pad_sequence(pred_mask, batch_first=True)

    # ---------- normalize ----------
    if args.dataset != 'ushcn':
        combined_vals = utils.normalize_masked_data(
            combined_vals, combined_mask, data_min, data_max)
        pred_val = utils.normalize_masked_data(
            pred_val, pred_mask, data_min, data_max)

    combined_tt = utils.normalize_masked_tp(combined_tt, 0, time_max)
    pred_tp     = utils.normalize_masked_tp(pred_tp,     0, time_max)

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "data_to_predict": pred_val,
        "tp_to_predict":  pred_tp,
        "mask_predicted_data": pred_mask,
    }

    # ---------- split into (B,M,L,D) ----------
    data_dict = utils.split_and_patch_batch(
        data_dict, args, n_obs_tp, patch_indices)
    return data_dict






if __name__ == '__main__':
	torch.manual_seed(1991)
	dataset = PhysioNet('../data/physionet', train=False, download=True)
	dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
	print(dataloader.__iter__().next())
