import os
import re
import pandas as pd
from datetime import datetime
import matplotlib
import logging
from Functions import \
    config_functions as cfg

matplotlib.use('agg')
logger = logging.getLogger(__name__)
cfg.set_logging()


# Run
if __name__ == '__main__':
    from Functions import \
        input_functions as ipt, \
        fit_functions as fit

    logger.info('\n==================================================================================='
                '\nCurrent run started at {}'
                '\n==================================================================================='
                ''.format(datetime.now()))

    edisgo_grid = 55
    charging_tech = 'G2V'
    max_parallel = 28
    basecost = True
    optimization = True
    testing = True

    # Set up
    filename = re.split('.py', os.path.split(os.path.abspath(__file__))[-1])[0]
    inputs = ipt.Inputs(edisgo_grid=edisgo_grid,
                        optimization=~basecost,
                        charging_tech=charging_tech,
                        filename=filename,
                        year=2035,
                        scenario=1,  # 4 = 0.05% PEV, 3 = 0% PEV
                        res_scenario=1)  # 0 = 0% RES
    inputs.dates_represented_index = inputs.index
    inputs.approach = 'budget'

    cost_all = pd.DataFrame(index=[], columns=cfg.best_cost_columns)
    for file in ['{}_cost_best.csv'.format(inputs.out_file),
                 '{}_cost_best_test.csv'.format(inputs.out_file)]:
        if not os.path.isfile(file):
            cost_all.to_csv(file)

    if basecost:
        fit.basecost_calc(inputs, max_parallel, invest_not_total=True)

    # Select worst PEV set and delete all others if requested
    inputs.charging_tech = charging_tech
    inputs.curtailment, inputs.optimize = True, True
    inputs.episode = inputs.update_worst_best_files(return_base=True, invest_not_total=True)
    inputs.update_names()  # Set worst PEV file as inputs.pev_file
    inputs.approach = 'budget'

    if optimization:
        fit.optimization(inputs, number_of_processes=max_parallel)
    if testing:
        fit.testing_calc(inputs, max_parallel, charging_tech=charging_tech, test_number=0, invest_not_total=True)

# pd.set_option('display.max_columns', None, 'display.max_rows', None)
# Memory optimization:
#  https://dzone.com/articles/python-memory-issues-tips-and-tricks
# Process information:
#  http://www.pybloggers.com/2016/02/psutil-4-0-0-and-how-to-get-real-process-memory-and-environ-in-python/
# Multiprocessing:
#  https://timber.io/blog/multiprocessing-vs-multithreading-in-python-what-you-need-to-know/
# Compiling options:
#  https://www.infoworld.com/article/2880767/5-projects-push-python-performance.html,
#  https://insights.dice.com/2018/06/28/4-fast-python-compilers-better-performance/,
#  https://www.nuitka.net/pages/overview.html, https://docs.python.org/2/library/py_compile.html,
#  https://doc.pypy.org/en/latest/introduction.html
