
import pandas as pd
import os
from tqdm import tqdm


def split_in_csv(temp, humidity, pres, root, type_meassures):

    spltting_temp = []
    spltting_hum = []
    spltting_pres = []
    add = 0
    # Creating dir for the samples
    try:
        os.mkdir(root + '/' + type_meassures + "/")
    except Exception as e:
        pass

    # splitting all in samples of length 15
    for i in tqdm(range(len(temp))):
        sample_to_generate = root + '/' + type_meassures
        sample_to_generate = sample_to_generate + \
            "/" + type_meassures + str(i) + '.csv'

        if os.path.isfile(sample_to_generate) == False:

            spltting_temp.append(temp[i])
            spltting_hum.append(humidity[i])
            spltting_pres.append(pres[i])
            add += 1

            if add == 15:
                df = pd.DataFrame()
                df[0] = spltting_temp
                df[1] = spltting_hum
                df[2] = spltting_pres
                add = 0
                spltting_temp = []
                spltting_hum = []
                spltting_pres = []
            # Saving the file
                df.to_csv(sample_to_generate, encoding='utf-8',
                          header=None, index=False)

            else:
                continue
