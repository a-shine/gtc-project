import pandas as pd

# TODO: Concat unites to the headers

# Standardize the column names and units



# df = pd.read_table('data/labquakes/Marone/p4581/p4581.txt', delim_whitespace=True, skiprows=[0, 1, 3, 4])
# df.to_csv('data/labquakes/Marone/p4581/p4581_processed.csv', index=False)
#
# df = pd.read_csv('data/labquakes/Marone/p4679/p4679.txt', skiprows=[1], usecols=[' LP_Disp', ' Shr_stress', ' nor_disp', ' Nor_stress', ' Time', ' mu', ' layer_thick', ' ec_disp'])
# df.columns = df.columns.str.strip()  # remove single white space from column names
# df.to_csv('data/labquakes/Marone/p4679/p4679_processed.csv', index=False)

df = pd.read_table('data/labquakes/MeleVeeduetal2020/b698/b698.txt', delim_whitespace=True, skiprows=[1], usecols=['lp_disp', 'LT', 'Tau', 'SigN', 'dcdtOB slip', 'Time', 'timedcdt', 'ec_disp', 'mu', 'Shear_Strain', 'Slip velocity'])
print(df.columns)
