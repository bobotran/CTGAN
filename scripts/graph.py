import numpy as np
import statistics
import matplotlib.pyplot as plt
import os
import pickle

#Identity_acc
#data = [0.825075, 0.824900, 0.826700, 0.827325, 0.828150, 0.826825, 0.824025, 0.828150, 0.823800, 0.827550, 0.827975, 0.825725, 0.827150, 0.824775, 0.823575, 0.824050, 0.827250, 0.826300, 0.825775, 0.826850, 0.824275, 0.826075, 0.825825, 0.827000, 0.823900, 0.830075, 0.824350]

#Identity_f1
#data = [0.661394, 0.654145, 0.664287, 0.668190, 0.668472, 0.667851, 0.660050, 0.659991, 0.668455, 0.665840, 0.665291, 0.668972, 0.667972, 0.665512, 0.660896, 0.661140, 0.665408, 0.666729, 0.666024, 0.667688, 0.662656, 0.664986, 0.667837, 0.667654, 0.658770, 0.665653, 0.664870]

#TrainBySampling_acc
#data = [0.676500, 0.646875, 0.658800, 0.689750, 0.736500, 0.673275, 0.692625, 0.652975, 0.738225, 0.629225, 0.711700, 0.642875, 0.656275, 0.669900, 0.691650, 0.649350, 0.594050, 0.644850, 0.684775, 0.660425, 0.710050, 0.673350, 0.659725, 0.649250, 0.690950, 0.666250, 0.693875, 0.692675, 0.682575, 0.671000]

#TrainBySampmling_f1
#data = [0.479381, 0.493247, 0.466624, 0.472080, 0.455275, 0.479426, 0.437954, 0.488211, 0.403875, 0.391368, 0.504568, 0.506104, 0.410702, 0.409581, 0.501057, 0.479877, 0.478212, 0.477470, 0.343051, 0.454623, 0.356201, 0.501401, 0.448958, 0.497825, 0.289587, 0.462472, 0.402755, 0.455701, 0.404548, 0.435766]

#Smote_acc
#data = [0.691875, 0.633350, 0.709525, 0.677725, 0.751675, 0.725050, 0.670275, 0.701150, 0.736200, 0.751875, 0.725425, 0.741100, 0.716725, 0.758625, 0.752475]

#Smote_f1
#data = [0.402425, 0.363202, 0.468345, 0.493379, 0.539800, 0.480882, 0.428438, 0.426571, 0.470387, 0.439758, 0.434143, 0.405149, 0.476293, 0.389735, 0.413839]

exp_name = 'baseline_likelihood'

metrics = ['syn_likelihood', 'test_likelihood']
for metric in metrics:
    fp = os.path.join('logs', exp_name, metric + '.pickle')
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    # Fit a normal distribution to the data:
    m = round(statistics.mean(data), 6)
    med = round(statistics.median(data), 6)
    std = round(statistics.stdev(data, m), 6)
    minimum = round(min(data), 6)
    maximum = round(max(data), 6)

    fp = os.path.join('logs', exp_name, 'metrics.txt')
    with open(fp, 'a') as f:
        log_str = '{} - '.format(metric)
        log_str += 'mean: {} '.format(m)
        log_str += 'median: {} '.format(med)
        log_str += 'std: {} '.format(std)
        log_str += 'min: {} '.format(minimum)
        log_str += 'max: {} '.format(maximum)
        log_str += '\n'
        f.write(log_str)
    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
    
    title = "{} {}: mean = {}, median = {}, std = {}, min = {}, max = {}".format(exp_name, metric, m, med, std, minimum, maximum)
    plt.title(title)
    
    plt.savefig(os.path.join('logs', exp_name, '{}_{}.png'.format(exp_name, metric)))
