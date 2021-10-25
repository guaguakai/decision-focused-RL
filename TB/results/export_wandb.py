import wandb
import os
import re
import pandas as pd


# Parse info from files and overwrite basic info if relevant
for folder in ['TS', 'DF']:
    for file in os.listdir(folder):
        # Log info on wandb
        run = wandb.init(reinit=True, project="DFRL-TB")
        wandb.run.name = file

        # Add basic info about config
        config = wandb.config
        config.discount = 0.9
        config.softness = 5
        config.warm_start = 1
        config.recycle = 0
        config.regularization = 0.1
        config.backprop_method = 1
        config.ess_const = 10
        config.number_trajectories = 100
        config.fully_observable = 1
        config.sample_size = 10

        config.method = folder

        # Parse filename
        params = re.findall(r'([a-zA-Z]+)([\d\.]+)', file)
        for param, val in params:
            if param == 'patients':
                config.num_patients = val
            elif param == 'demo':
                config.demonstrate_softness = val
            elif param == 'noise':
                config.noise = val
            elif param == 'actioneffect':
                config.effect_size = val
            elif param == 'startadhering':
                config.start_adhering = val
            elif param == 'seed':
                config.seed = val
            else:
                print(f"Unrecognized Parameter {param} ({val})")

        # Parse file
        df = pd.read_csv(os.path.join(folder, file))

        # Log file
        for log in df.to_dict('records'):
            if log['epoch'] >= 0:
                wandb.log(log)

        run.finish()
