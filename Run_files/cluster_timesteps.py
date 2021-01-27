
from datetime import datetime
import matplotlib
import logging
from Functions import \
    config_functions as cfg

matplotlib.use('agg')
logger = logging.getLogger(__name__)
cfg.set_logging()


def cluster(grid):
    import os
    import re
    from Functions import \
        input_functions as ipt, grid_functions as grd

    logger.info('Working on grid: {}'.format(grid))

    # Set up
    filename = re.split('.py', os.path.split(os.path.abspath(__file__))[-1])[0]
    inputs = ipt.Inputs(edisgo_grid=grid,
                        optimization=False,
                        charging_tech='G2V',
                        filename=filename,
                        year=2035,
                        scenario=1,  # 4 = 0.05% PEV, 3 = 0% PEV
                        res_scenario=1)  # 0 = 0% RES
    inputs.dates_represented_index = inputs.index
    inputs.approach = 'budget'
    grid = grd.get_grid(inputs)

    logger.info('Grid {} imported'.format(grid))
    start = 2
    stop = 2000
    steps = 50
    grid.random_sample = cfg.timestep_samples(inputs, grid,
                                              elbow_plot=True,
                                              visualize=False,
                                              start=start,
                                              stop=stop,
                                              steps=steps)

# Run
if __name__ == '__main__':
    import multiprocessing as mp
    from functools import partial

    logger.info('\n==================================================================================='
                '\nCurrent run started at {}'
                '\n==================================================================================='
                ''.format(datetime.now()))

    edisgo_grids = [163, 404, 55, 1055, 1392, 1791, 1944, 2948]
    pool = mp.Pool(processes=len(edisgo_grids),
                   maxtasksperchild=1)
    func = partial(cluster)
    pool.map(func, edisgo_grids)
    pool.close()
    pool.join()

