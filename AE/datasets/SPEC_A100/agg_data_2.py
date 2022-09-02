# from tkinter.tix import Tree
from typing_extensions import runtime
from unittest import skip
import pandas as pd
import numpy as np
import codecs

import sys

def accel_split_perf(arch):
    apps=["tpacf", "stencil", "lbm", "fft", "spmv", "mriq", "histo", "bfs", "cutcp", "kmeans", "lavamd", "cfd", "nw", "hotspot", "lud", "ge", "srad", "heartwall", "bplustree"]
    df = pd.read_csv(arch+'-dvfs-accel-perf.csv')
    df = df.drop(columns=['Unnamed: 0'])
    for app in apps:
        app_df = df.loc[df['application'] == app]
        app_df = app_df.copy()
        app_df.reset_index(inplace=True,drop=True)
        app_df.to_csv(arch+'-dvfs-'+app+'-perf.csv')

# arch = 'GA100'
# accel_split_perf(arch)
# sys.exit(0)

def accel_perf_data(p,arch):
    apps=["tpacf", "stencil", "lbm", "fft", "spmv", "mriq", "histo", "bfs", "cutcp", "kmeans", "lavamd", "cfd", "nw", "hotspot", "lud", "ge", "srad", "heartwall", "bplustree"]
    # freqs=[1380, 1372, 1365, 1357, 1350, 1342, 1335, 1327, 1320, 1312, 1305, 1297, 1290, 1282, 1275, 1267, 1260, 1252, 1245, 1237, 1230, 1222, 1215, 1207, 1200, 1192, 1185, 1177, 1170, 1162, 1155, 1147, 1140, 1132, 1125, 1117, 1110, 1102, 1095, 1087, 1080, 1072, 1065, 1057, 1050, 1042, 1035, 1027, 1020, 1012, 1005, 997, 990, 982, 975, 967, 960, 952, 945, 937, 930, 922, 915, 907, 900, 892, 885, 877, 870, 862, 855, 847, 840, 832, 825, 817, 810, 802, 795, 787, 780, 772, 765, 757, 750, 742, 735, 727, 720, 712, 705, 697, 690, 682, 675, 667, 660, 652, 645, 637, 630, 622, 615, 607, 600, 592, 585, 577, 570, 562, 555, 547, 540, 532, 525, 517, 510, 502, 495, 487, 480, 472, 465, 457, 450, 442, 435, 427, 420, 412, 405]
    freqs = ['1410', '1395', '1380', '1365', '1350', '1335', '1320', '1305', '1290', '1275', '1260', '1245', '1230', '1215', '1200', '1185', '1170', '1155', '1140', '1125', '1110', '1095', '1080', '1065', '1050', '1035', '1020', '1005', '990', '975', '960', '945', '930', '915', '900', '885', '870', '855', '840', '825', '810', '795', '780', '765', '750', '735', '720', '705', '690', '675', '660', '645', '630', '615', '600', '585', '570', '555', '540', '525', '510']
    i=1 
    accel_perf = {"base_kernel_runtime":[],"kernel_runtime":[],"ratio":[],"dvfs":[],"application":[]}

    for freq in freqs:
        for r in range (3):
            for app in apps:
                if i >= 100:
                    index = str(i)
                elif i < 10:
                    index = '00'+str(i)
                elif i < 100:
                    index = '0'+str(i)

                file=p+"ACCEL_OCL."+index+".opencl.ref.csv"
                f = open(file, "r")
                for l in f:
                    index = l.find(app)
                    if index != -1:
                        d = l.split(',')
                        if len (d) < 3:
                            print ('**** NOT RUNTIME LINE ****')
                            continue
                        accel_perf["base_kernel_runtime"].append(round(float(d[1]),2))
                        accel_perf["kernel_runtime"].append(round(float(d[2]),2))
                        accel_perf["ratio"].append(round(float(d[3]),2))
                        accel_perf["dvfs"].append(freq)
                        accel_perf["application"].append(app)
                        break
                         
                i = i + 1
    df = pd.DataFrame.from_dict(accel_perf)
    df = df.groupby(['dvfs','application'],as_index=False).mean().reset_index(inplace=False,drop=True)
    print (df)
    df.to_csv(arch+'-dvfs-accel-perf.csv')
# path = 'spec_time_results/'
# arch = 'GA100'
# accel_perf_data(path,arch)
# sys.exit(0)

def agg_app_perf_data(app,arch):
    df = pd.read_csv(arch+'-dvfs-'+app+'-perf.csv')
    df.columns =['dvfs', 'flops', 'kernel_runtime']
    df = df.loc[df['dvfs'] >= 510]
    df = df.groupby('dvfs').mean().reset_index()
    df.to_csv(arch+'-dvfs-'+app+'-perf.csv')
    print(df)

'''
arch = 'GA100'
agg_app_perf_data('dgemm',arch)
# agg_app_perf_data('stream',arch)
# agg_app_perf_data('lstm',arch)
# agg_app_perf_data('namd',arch)
# agg_app_perf_data('lammps',arch)
sys.exit(0)
'''
def avg_val(lst):
    return (sum(lst) / len(lst))

def fill_with_previous(df):
    return df.replace(to_replace=0, method='ffill')

def agg_dcgm_data(app,arch,p):

    dvfsl = []
    dramal = []
    fp64al = []
    fp32al = []
    powerl = []
    energyl = []
    GRACT_l = []  
    SMACT_l = []  
    SMOCC_l = []    
    PCITX_l = []                 
    PCIRX_l = []       
    GPUTL_l = []       
    MCUTL_l = []       
    MUSAM_l = []       
    GUSAM_l = []
    TENSO_l = []

    dram_list = []
    fp64a_list = []
    fp32a_list = []
    power_list = []
    energy_list = []
    GRACT_list = []  
    SMACT_list = []  
    SMOCC_list = []    
    PCITX_list = []                 
    PCIRX_list = []       
    GPUTL_list = []       
    MCUTL_list = []       
    MUSAM_list = []       
    GUSAM_list = []
    TENSO_list = []
   
    runs_df = []
    agg_df = []

    # , '502', '495', '487', '480', '472', '465', '457', '450', '442', '435', '427', '420', '412', '405'
    # freqs = ['1380', '1372', '1365', '1357', '1350', '1342', '1335', '1327', '1320', '1312', '1305', '1297', '1290', '1282', '1275', '1267', '1260', '1252', '1245', '1237', '1230', '1222', '1215', '1207', '1200', '1192', '1185', '1177', '1170', '1162', '1155', '1147', '1140', '1132', '1125', '1117', '1110', '1102', '1095', '1087', '1080', '1072', '1065', '1057', '1050', '1042', '1035', '1027', '1020', '1012', '1005', '997', '990', '982', '975', '967', '960', '952', '945', '937', '930', '922', '915', '907', '900', '892', '885', '877', '870', '862', '855', '847', '840', '832', '825', '817', '810', '802', '795', '787', '780', '772', '765', '757', '750', '742', '735', '727', '720', '712', '705', '697', '690', '682', '675', '667', '660', '652', '645', '637', '630', '622', '615', '607', '600', '592', '585', '577', '570', '562', '555', '547', '540', '532', '525', '517', '510']
    # freqs = ['1380']
    freqs = ['1410', '1395', '1380', '1365', '1350', '1335', '1320', '1305', '1290', '1275', '1260', '1245', '1230', '1215', '1200', '1185', '1170', '1155', '1140', '1125', '1110', '1095', '1080', '1065', '1050', '1035', '1020', '1005', '990', '975', '960', '945', '930', '915', '900', '885', '870', '855', '840', '825', '810', '795', '780', '765', '750', '735', '720', '705', '690', '675', '660', '645', '630', '615', '600', '585', '570', '555', '540', '525', '510']
    apps = ["tpacf", "stencil", "lbm", "fft", "spmv", "mriq", "histo", "bfs", "cutcp", "kmeans", "lavamd", "cfd", "nw", "hotspot", "lud", "ge", "srad", "heartwall", "bplustree"]
    # apps = [app] 
    datalists = []
    # apps = ['lbm']
    ttt = 0
    rs = ['-0','-1','-2']
    for app in apps:
        rt = pd.read_csv(arch+"-dvfs-"+app.lower()+"-perf.csv")
        for f in freqs:
            for r in rs:
                # GV100-dvfs-NAMD-997-2
                f1 = p+arch+"-"+app+"-"+f+r # FOR SPEC FILES
                # f1 = p+arch+"-dvfs-"+app+"-"+f+r # FOR DGEMM FILES
                # reading data
                # df1 = pd.read_csv(f1,delim_whitespace=True,error_bad_lines=False) 
                # df1 = pd.read_csv(f1, delim_whitespace=True,error_bad_lines=False, engine ='python')
                # doc = codecs.open(f1,'rU','UTF-16') #open for reading with "universal" type set
                # df1 = pd.read_csv(doc, sep='\t')

                df = pd.read_csv(f1,engine="python")
                cols = df.columns
                columns = ' '.join(cols).split()
                columns.remove('#')
                columns.insert(1,'ID')
                # print(columns)
                for index, row in df.iterrows():
                        lst = row[0].split(" ")
                        arow = ' '.join(lst).split()
                        if ('mJ' in arow) | ('GRACT' in arow):
                        #     print ('YES mJ')
                            continue
                        datalists.append(arow)
                        # dataindex = 0
                        # for v in arow:
                        #     data[columns[dataindex]] = [v]
                        #     dataindex = dataindex + 1 
                
                df1 = pd.DataFrame(datalists,columns=columns)
                datalists.clear()
                
                # conversion from string to numeric
                df1["TENSO"] = pd.to_numeric(df1["TENSO"], downcast="float")
                df1["FP16A"] = pd.to_numeric(df1["FP16A"], downcast="float")
                df1["FP32A"] = pd.to_numeric(df1["FP32A"], downcast="float")
                df1["FP64A"] = pd.to_numeric(df1["FP64A"], downcast="float")
                df1["DRAMA"] = pd.to_numeric(df1["DRAMA"], downcast="float")
                df1["SACLK"] = pd.to_numeric(df1["SACLK"])
                df1["POWER"] = pd.to_numeric(df1["POWER"], downcast="float")
                df1["TOTEC"] = pd.to_numeric(df1["TOTEC"])
                df1["TOTEC"] = ((df1['TOTEC'].iloc[-1] - df1['TOTEC'].iloc[0])/1000)
                df1["GRACT"] = pd.to_numeric(df1["GRACT"], downcast="float")  
                df1["SMACT"] = pd.to_numeric(df1["SMACT"], downcast="float")  
                df1["SMOCC"] = pd.to_numeric(df1["SMOCC"], downcast="float")    
                df1["PCITX"] = pd.to_numeric(df1["PCITX"])                 
                df1["PCIRX"] = pd.to_numeric(df1["PCIRX"])       
                df1["GPUTL"] = pd.to_numeric(df1["GPUTL"], downcast="float")       
                df1["MCUTL"] = pd.to_numeric(df1["MCUTL"], downcast="float")       
                df1["MUSAM"] = pd.to_numeric(df1["MUSAM"], downcast="float")       
                df1["GUSAM"] = pd.to_numeric(df1["GUSAM"], downcast="float")
                df1["PSTAT"] = pd.to_numeric(df1["PSTAT"])
                df1["TMPTR"] = pd.to_numeric(df1["TMPTR"], downcast="float")
                df1["MMTMP"] = pd.to_numeric(df1["MMTMP"], downcast="float")
                df1["SMCLK"] = pd.to_numeric(df1["SMCLK"])
                df1["ID"] = pd.to_numeric(df1["ID"])

                # filtering and only take rows where there is an activity
                data = df1.loc[(df1['FP64A'] > 0) | (df1['FP32A'] > 0) | (df1['TENSO'] > 0)]
                # print (data)    
                if data.shape[0] == 0:
                    print(app,f, r,'**************** No FP Activity recorded ****************')
                    continue
                elif round(data['DRAMA'].mean(),3) == 0:
                    print('NO DRAMA ACTIVITY')
                    # continue
                elif (round(data['FP64A'].mean(),3) == 0) & (round(data['FP32A'].mean(),3) == 0) & (round(data['TENSO'].mean(),3) == 0):
                    print('NO FP ACTIVITY')
                    continue
                # elif round(data['FP32A'].mean(),3) == 0:
                    # print('')
                    # continue
                # take averages and append to lists
                # print(f,)
                df = fill_with_previous(data)
                print(app, r,f,'saved data')
                runs_df.append(df)
            r3_df = pd.concat(runs_df, ignore_index=True)
            mean_df = r3_df.groupby('SACLK').mean()
            
            runtime = rt.loc[rt['dvfs'] == int(f)]['kernel_runtime'].values[0]
            if app == 'dgemm':
                runtime /= 1000

            print (mean_df)
            print(mean_df.shape[0])
            mean_df["run_time"] = [runtime]
            mean_df["application"] = [app]
            mean_df['dvfs'] = [int(f)]
            agg_df.append(mean_df)
            runs_df.clear()
            # mean_df.drop(mean_df.index, inplace=True)
            # print (mean_df)
            # dramal = [i for i in dramal if i != 0]
            # fp64al = [i for i in fp64al if i != 0]
            # powerl = [i for i in powerl if i != 0]
            # energyl = [i for i in energyl if i != 0]
            
            # print (app,f,'dram:', dram_list)
            # print (f,'fp64:', fp64a_list)
            # print (f,'fp32:', fp32a_list)
            # print (f,'power:', power_list)
            # print (f,'energy:', energy_list)
    print ('agg length',len(agg_df))
    merged_df = pd.concat(agg_df)
    # print(merged_df)
    merged_df.rename(columns = {'TOTEC':'total_energy_consumption','GUSAM':'gpu_util_samples','MUSAM':'mem_util_samples','MCUTL':'mem_copy_utilization','TENSO':'tensor_activity','SMOCC':'sm_occupancy','GRACT':'gr_engine_active','SMACT':'sm_active','PCITX':'pcie_tx_bytes','PCIRX':'pcie_rx_bytes','GPUTL':'gpu_utilization','dvfs':'sm_app_clock','FP16A':'fp16_active','FP32A':'fp32_active','FP64A':'fp64_active','POWER':'power_usage','DRAMA':'dram_active','SMCLK':'sm_clock','PSTAT':'pstate','TMPTR':'gpu_temp','MMTMP':'memory_temp'}, inplace = True)
    merged_df.reset_index(drop=True,inplace=True)
    print(merged_df)
    # print ('*** RAW DATA ***',raw_data)        
    merged_df.to_csv(arch+'-dvfs-accel-dcgm.csv')
    

# path = '/mnt/c/rf/dcgmi/accel/'
# path = '/home/ghali/gpu/results/'
# path = '/home/ghali/gpu/hpc_apps_v2/LAMMPS/LJ'
app = 'dgemm' #'stream'#'NAMD'#'LSTM'#'LAMMPS'
# agg_app_perf_data(app.lower())
arch = 'GA100'
# arch = 'GV100'
path = 'dcgm_results/'
agg_dcgm_data(app,arch,path)