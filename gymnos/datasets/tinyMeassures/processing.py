
import pandas as pd
import os
from tqdm import tqdm

def split_in_csv(temp,humidity,pres,root,type_meassures):

    spltting_temp = []
    spltting_hum = []
    spltting_pres = []
    add = 0
    try:
        os.mkdir(root +'/'+type_meassures+"/")
    except Exception as e:
        pass

    for i in tqdm(range(len(temp))):

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
            df.to_csv(root+'/'+type_meassures+"/"+type_meassures+str(i)+'.csv',encoding='utf-8',  header=None,index=False)
