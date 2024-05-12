import argparse
import subprocess
import re
import torch_fidelity
import datetime
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_info(*args):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{dtstr}]", *args)

def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
    """
    Get estimated total time and ETA time.
    :param time_start:
    :param elapsed_iter:
    :param total_iter:
    :return: string of elapsed time, string of ETA
    """

    def sec_to_str(sec):
        val = int(sec)  # seconds in int type
        s = val % 60
        val = val // 60  # minutes
        m = val % 60
        val = val // 60  # hours
        h = val % 24
        d = val // 24  # days
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"

    elapsed_time = time.time() - time_start  # seconds elapsed
    elp = sec_to_str(elapsed_time)
    if elapsed_iter == 0:
        eta = 'NA'
    else:
        # seconds
        eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
        eta = sec_to_str(eta)
    return elp, eta

def calc_fid(gpu, fid_subprocess: bool, input1, input2, logger=log_info):
    if fid_subprocess:
        cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
        logger(f"cmd: {cmd}")
        cmd_arr = cmd.split(' ')
        res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
        output = str(res.stdout)
        logger(f"out: {output}")  # frechet_inception_distance: 16.5485\n
        m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
        fid = float(m.group(1))
    else:
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=input1,
            input2=input2,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
            samples_find_deep=True,
        )
        fid = metrics_dict['frechet_inception_distance']
    return fid

def calc_fid_isc(gpu, input1, input2, isc_flag=True, logger=log_info):
    if isc_flag:
        cmd = f"fidelity --gpu {gpu} --fid --isc --input1 {input1} --input2 {input2} --silent"
    else:
        cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
    logger(f"cmd: {cmd}")
    cmd_arr = cmd.split(' ')
    res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
    output = str(res.stdout)
    # inception_score_mean: 11.24431
    # inception_score_std: 0.09522244
    # frechet_inception_distance: 71.2743
    logger(f"out: {output}")
    m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
    fid = float(m.group(1))
    if isc_flag:
        m = re.search(r'inception_score_mean: (\d+\.\d+)', output)
        is_mean = float(m.group(1))
        m = re.search(r'inception_score_std: (\d+\.\d+)', output)
        is_std = float(m.group(1))
    else:
        is_mean, is_std = 0., 0.
    return fid, is_mean, is_std

def get_model_id_by_name(name):
    d = {
        'sd15'          : "runwayml/stable-diffusion-v1-5",
        '2rf_sd15'      : "XCLIU/2_rectified_flow_from_sd_1_5",
        'instaflow_09b' : "XCLIU/instaflow_0_9B_from_sd_1_5",
    }
    return d.get(name, None)
