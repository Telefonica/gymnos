
import pandas as pd
import os

def split_in_csv(temp,humidity,pres,download_dir,type_meassures):

    spltting_temp = []
    spltting_hum = []
    spltting_pres = []
    add = 0
    os.mkdir(download_dir+'/'+type_meassures+"/")
    for i in range(len(temp)):
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
            df.to_csv(download_dir+'/'+type_meassures+"/"+type_meassures+str(i)+'.csv',encoding='utf-8',  header=None,index=False)
