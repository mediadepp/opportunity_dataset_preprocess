# IHSN 

import logging 
import numpy as np 
import os 
import pandas as pd 
join = os.path.join 
logging.basicConfig(
    filename="logger.log", 
    filemode="w", 
    format="%(name)s - %(levelname)s - %(message)s", 
    level=logging.DEBUG, 
) 


def _get_column_names(column_line, sensor_names): 
    for sensor in sensor_names: 
        if (sensor in column_line) and ("Quaternion" not in column_line): 
            return sensor 
    return None 


def columns_to_keep(data_columns, label_columns): 
    result = [] 
    names = [] 
    result += [int(data_columns[0][2])-1] 
    names += [data_columns[0][-1]] 
    for column in data_columns[1:]: 
        result += [int(column[1])-1] 
        names += [f"{column[-2]} - {column[-1]}"] 
    for label in label_columns: 
        result += [int(label[0])-1] 
        names += [label[-1]] 
    return result, names 


def columns_to_remove(
    indices_to_keep, 
    number_of_columns, 
): 
    all_indices = set(list(range(number_of_columns))) 
    rm_indices = all_indices - set(indices_to_keep) 
    rm_indices = list(sorted(rm_indices, key=lambda x: x, reverse=False)) 
    return rm_indices 


def drop_columns(data, columns_to_drop): 
    res = np.delete(data, obj=columns_to_drop, axis=1)
    return res 


def get_column_details(sensor_path, column_detail_path): 
    sensors = [] 
    with open(sensor_path, mode="r", encoding="utf8") as f: 
        for line in f: 
            content = line.strip(", \n") 
            if "#" not in content and len(content) > 0: 
                sensors += [content] 
    column_details = [] 
    label_mode = False 
    labels = [] 
    with open(column_detail_path, mode="r", encoding="utf8") as f: 
        for line in f:  
            content = line.strip() 
            if "label" in content.lower():
                label_mode = True 
            if content.lower().startswith("column"): 
                sensor = _get_column_names(content, sensors) 
                if (sensor is not None) and (label_mode == False): 
                    column_detail = content.strip().split(";")[0].split()[-4:] 
                    column_detail = [sensor] + column_detail 
                    column_details += [column_detail] 
                if label_mode: 
                    label = content.strip().split(" ")[-2:] 
                    labels += [label] 
    for column_detail in column_details: 
        logging.debug(f"{column_detail}") 
    for label in labels: 
        logging.debug(f"{label}") 
    return column_details, labels 

    
def get_data(
    addrs, 
): 
    raw_data = np.loadtxt(addrs) 
    return raw_data 


def get_all_data(
    root, 
    general_frmt=lambda s, f: f"S{s}-ADL{f}.dat", 
    drill_frmt=lambda s: f"S{s}-Drill.dat", 
    num_of_users=4, 
    num_of_data_per_user=5, 
    use_drill=True, 
): 
    general_addrs = join(root, general_frmt()) 
    drill_addrs = join(root, drill_frmt()) 
    for user in range(num_of_users): 
        for data_num in range(num_of_data_per_user): 
            pass 



# def get_cleaned_user_data(file_pth):
#     """Remove unused sensor modes, remove active conversion data, 
#     and interpolate to complete NaN data
#     """
#     invalid_feature = np.arange(46, 50)  # BACK Quaternion
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(34, 37)] )  # RH_acc
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(59, 63)] )  # RUA Quaternion
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(72, 76)] )  # RLA
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(85, 89)] )  # LUA
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(99, 102)] )  # LLA
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(117, 118)] )  # L-SHOE Compass
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(133, 134)] )  # R-SHOE Compass
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(134, 244)] )  # environment sensor
#     invalid_feature = np.concatenate( [invalid_feature, np.arange(245, 250)] )  # LL, ML level label
#     drop_columns = invalid_feature
#     logging.debug(f"drop columns: {drop_columns}, len of drop_columns: {len(drop_columns)}")
#     raw_data = np.loadtxt(file_pth) 
#     logging.debug(f"raw_data: {raw_data.shape}")
#     logging.debug(f"raw_data type: {type(raw_data)}")
#     used_data = np.delete(raw_data, drop_columns, axis=1)
#     logging.debug(f"after dropping: {used_data.shape}") 
#     
#     used_columns = ["MILLISEC", "acc_RKN_upper_accX","acc_RKN_upper_accY","acc_RKN_upper_accZ",
#                     "acc_HIP_accX","acc_HIP_accY","acc_HIP_accZ",
#                     "acc_LUA_upper_accX","acc_LUA_upper_accY","acc_LUA_upper_accZ",
#                     "acc_RUA_lower_accX","acc_RUA_lower_accY","acc_RUA_lower_accZ",
#                     "acc_LH_accX","acc_LH_accY","acc_LH_accZ",
#                     "acc_BACK_accX","acc_BACK_accY","acc_BACK_accZ",
#                     "acc_RKN_lower_accX","acc_RKN_lower_accY","acc_RKN_lower_accZ",
#                     "acc_RWR_accX","acc_RWR_accY","acc_RWR_accZ",
#                     "acc_RUA_upper_accX","acc_RUA_upper_accY","acc_RUA_upper_accZ",
#                     "acc_LUA_lower_accX","acc_LUA_lower_accY","acc_LUA_lower_accZ",
#                     "acc_LWR_accX","acc_LWR_accY","acc_LWR_accZ",
# #                     "acc_RH_accX","acc_RH_accY","acc_RH_accZ",
#                     "imu_BACK_accX","imu_BACK_accY","imu_BACK_accZ",
#                     "imu_BACK_gyroX","imu_BACK_gyroY","imu_BACK_gyroZ",
#                     "imu_BACK_magneticX","imu_BACK_magneticY","imu_BACK_magneticZ",
#                     "imu_RUA_accX","imu_RUA_accY","imu_RUA_accZ",
#                     "imu_RUA_gyroX","imu_RUA_gyroY","imu_RUA_gyroZ",
#                     "imu_RUA_magneticX","imu_RUA_magneticY","imu_RUA_magneticZ",
#                     "imu_RLA_accX","imu_RLA_accY","imu_RLA_accZ",
#                     "imu_RLA_gyroX","imu_RLA_gyroY","imu_RLA_gyroZ",
#                     "imu_RLA_magneticX","imu_RLA_magneticY","imu_RLA_magneticZ",
#                     "imu_LUA_accX","imu_LUA_accY","imu_LUA_accZ",
#                     "imu_LUA_gyroX","imu_LUA_gyroY","imu_LUA_gyroZ",
#                     "imu_LUA_magneticX","imu_LUA_magneticY","imu_LUA_magneticZ",
#                     "imu_LLA_accX","imu_LLA_accY","imu_LLA_accZ",
#                     "imu_LLA_gyroX","imu_LLA_gyroY","imu_LLA_gyroZ",
#                     "imu_LLA_magneticX","imu_LLA_magneticY","imu_LLA_magneticZ",
#                     "imu_L-SHOE_EuX","imu_L-SHOE_EuY","imu_L-SHOE_EuZ",
#                     "imu_L-SHOE_Nav_Ax","imu_L-SHOE_Nav_Ay","imu_L-SHOE_Nav_Az",
#                     "imu_L-SHOE_Body_Ax","imu_L-SHOE_Body_Ay","imu_L-SHOE_Body_Az",
#                     "imu_L-SHOE_AngVelBodyFrameX","imu_L-SHOE_AngVelBodyFrameY","imu_L-SHOE_AngVelBodyFrameZ",
#                     "imu_L-SHOE_AngVelNavFrameX","imu_L-SHOE_AngVelNavFrameY","imu_L-SHOE_AngVelNavFrameZ",
#                     "imu_R-SHOE_EuX","imu_R-SHOE_EuY","imu_R-SHOE_EuZ",
#                     "imu_R-SHOE_Nav_Ax","imu_R-SHOE_Nav_Ay","imu_R-SHOE_Nav_Az",
#                     "imu_R-SHOE_Body_Ax","imu_R-SHOE_Body_Ay","imu_R-SHOE_Body_Az",
#                     "imu_R-SHOE_AngVelBodyFrameX","imu_R-SHOE_AngVelBodyFrameY","imu_R-SHOE_AngVelBodyFrameZ",
#                     "imu_R-SHOE_AngVelNavFrameX","imu_R-SHOE_AngVelNavFrameY","imu_R-SHOE_AngVelNavFrameZ",
#                     "Locomotion",
#                     "HL_Activity"]
#     used_data = pd.DataFrame(used_data, columns=used_columns)
#     logging.debug(f"used data (dataframe shape): {used_data.shape}") 
#     used_data = used_data[used_data['HL_Activity'] != 0]  # Activity conversion data label is 0, discarded
#     logging.debug(f"after discarding some rows: {used_data.shape}")
# 
#     used_data['HL_Activity'][used_data['HL_Activity']==101] = 0  # Relaxing
#     used_data['HL_Activity'][used_data['HL_Activity']==102] = 1  # Coffee time
#     used_data['HL_Activity'][used_data['HL_Activity']==103] = 2  # Early morning
#     used_data['HL_Activity'][used_data['HL_Activity']==104] = 3  # Cleanup
#     used_data['HL_Activity'][used_data['HL_Activity']==105] = 4  # Sandwich time
#     
#     logging.debug(f"before interpolation: {used_data.shape}")
#     used_data = used_data.interpolate()
#     logging.debug(f"after interpolation: {used_data.shape}")
# 
#     # View the location of Nan data
#     pos = used_data.isnull().stack()[lambda x:x].index.tolist()
#     logging.debug(f"View the location of Nan data: {pos}. ")
#     
#     used_data = used_data.dropna(axis=0)
#     logging.debug(f"after dropping: {used_data.shape}")
#     return used_data





if __name__ == "__main__": 
    root = r"/home/madmedi/Documents/papers/HAR/my_code/1_opportunity/opportunity/dataset" 
    # file_names = [] 
    # for name in os.listdir(root): 
    #     file_names += [join(root, name)] 
    #     logging.debug(msg=f"{file_names[-1]}") 
    # number_of_users = 4 
    # number_of_samples = 5 
    # frmt = lambda s, f: f"S{s}-ADL{f}.dat" 
    # data = {} 
    # for user in range(number_of_users): 
    #     data[user+1] = [] 
    #     for sample in range(number_of_samples): 
    #         file_name = frmt(user+1, sample+1) 
    #         file_path = join(root, file_name) 
    #         data[user+1] += [file_path] 
    # logging.debug("\n") 
    # logging.debug(data) 
    # get_cleaned_user_data(data[1][0]) 
    data_columns, label_columns = get_column_details(
        sensor_path=r"./sensor_names.txt", 
        column_detail_path=f"{join(root, 'column_names.txt')}", 
    ) 
    user_id = 3 
    turn = 0 
    data = get_data(
        addrs=join(root, (lambda s, f: f"S{s}-ADL{f}.dat")(user_id+1, turn+1)) 
    ) 
    logging.debug(f"data: {data.shape}")
    indices_to_keep, names_indices_to_keep = columns_to_keep(
        data_columns=data_columns, 
        label_columns=label_columns, 
    )
    logging.debug(f"Columns to keep: {indices_to_keep}")
    logging.debug("==================================================")
    logging.debug(f"Columns to keep: {names_indices_to_keep}")
    indices_to_remove = columns_to_remove(
        indices_to_keep=indices_to_keep, 
        number_of_columns=data.shape[1], 
    )
    logging.debug(indices_to_remove) 
    logging.debug("\n") 
    data = drop_columns(
        data=data, 
        columns_to_drop=indices_to_remove, 
    )
    logging.debug(data.shape) 
    assert len(indices_to_keep) == data.shape[1] 
    assert len(indices_to_keep) == len(names_indices_to_keep) 
    df = pd.DataFrame(data, columns=names_indices_to_keep) 
    logging.debug(df.shape)
    logging.debug(df.head()) 
    logging.debug("Loco")
    logging.debug(df['Locomotion'].value_counts())
    logging.debug("HL_Activity")
    logging.debug(df['HL_Activity'].value_counts())
    logging.debug("LL_Left_Arm")
    logging.debug(df['LL_Left_Arm'].value_counts())
    logging.debug("LL_Left_Arm_Object")
    logging.debug(df['LL_Left_Arm_Object'].value_counts())
    logging.debug("LL_Right_Arm")
    logging.debug(df['LL_Right_Arm'].value_counts())
    logging.debug("LL_Right_Arm_Object")
    logging.debug(df['LL_Right_Arm_Object'].value_counts())
    logging.debug("ML_Both_Arms")
    logging.debug(df['ML_Both_Arms'].value_counts())
