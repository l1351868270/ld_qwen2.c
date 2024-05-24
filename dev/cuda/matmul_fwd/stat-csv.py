# refer to https://github.com/reed-lau/cute-gemm/blob/main/stat-csv.py
from pprint import pprint
import numpy as np

def load_csv_and_stat(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()

    ret = {} 
    include_funcs = set([
        'ampere_h1688gemm_128x128_ldg8_stages_32x1_tn',
        'ampere_h16816gemm_128x64_ldg8_stages_32x6_tn',
        "ampere_h1688gemm_256x64_ldg8_stages_32x1_tn",
        "cutlass_80_tensorop_s16816gemm_f16_64x64_32x6_tn_align8",
        'simple_gemm_tt',
    ])
    for line in lines:
        if line.startswith('=='):
            continue
        if line.startswith(r'"ID"'):
            continue

        fields = line.split(',\"')
        kernel = fields[4].replace('"', '')
        # try:
        #   usec = float(fields[-1].replace('"', '').replace(',', '.'))
        usec = float(fields[-1].replace('"', '').replace(',', ''))
        # except:
        #   continue
        is_save = False
        func_name = ''
        for func in include_funcs:
            if func in kernel:
                is_save = True
                func_name = func

        if is_save:
            if func_name not in ret:
                ret[func_name] = [] # usec
            ret[func_name].append(usec)
            # print(f'kernel: {kernel}, usec: {usec}')

    print('kernel, mean(us), std, med, num')
    for k, s in ret.items():
        s = np.array(s) / 1000.0
        num = len(s)
        mean = np.mean(s)
        std = np.std(s)
        med = np.median(s)
        ret[k] = (mean, std, med, num) 
        print(f'{k},  {mean:.3f},  {std:.3f},  {med:.3f},  {num}')
    # pprint(ret)

load_csv_and_stat('benchmark.csv')
