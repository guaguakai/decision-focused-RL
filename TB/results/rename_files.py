import os
import re
import pdb


for folder in ['TS', 'DF']:
    for file in os.listdir(folder):
    	# Make sure the file has a given prefix
    	if '0524' not in file:
    		continue

    	# Remove the '-seed{}'
    	new_file = re.sub(r'-seed\d+', '', file)

    	# Change the filename
    	os.rename(os.path.join(folder, file), os.path.join(folder, new_file))
