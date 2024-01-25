# IHSN

from collections import defaultdict
import logging
import math
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import windower

join = os.path.join
logging.basicConfig(
    filename="data_logging.log",
    filemode="w",
    format="%(name)s - %(lineno)d - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


def _get_column_names(column_line, sensor_names):
    insufficient = [
        "RH accX",
        "RH accY",
        "RH accZ",
        "Quaternion1",
        "Quaternion2",
        "Quaternion3",
        "Quaternion4",
        "L-SHOE Compass",
        "R-SHOE Compass",
    ]
    for sensor in sensor_names:
        if (sensor in column_line) and len(
            [ins for ins in insufficient if ins not in column_line]
        ) == len(insufficient):
            return sensor
    return None


def _columns_to_keep(data_columns, label_columns):
    result = []
    names = []
    result += [int(data_columns[0][2]) - 1]
    names += [data_columns[0][-1]]

    for column in data_columns[1:]:
        result += [int(column[1]) - 1]
        names += [f"{column[-3]} - {column[-2]} - {column[-1]}"]

    for label in label_columns:
        result += [int(label[0]) - 1]
        names += [label[-1]]

    return result, names


def _columns_to_remove(
    indices_to_keep,
    number_of_columns,
):
    all_indices = set(list(range(number_of_columns)))
    rm_indices = all_indices - set(indices_to_keep)
    rm_indices = list(sorted(rm_indices, key=lambda x: x, reverse=False))
    return rm_indices


def _drop_columns(data, columns_to_drop):
    res = np.delete(data, obj=columns_to_drop, axis=1)
    return res


def _get_column_details(sensor_path, column_detail_path):
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

    if False:
        logging.debug(f"input features ({len(column_details)}): ")
        for column_detail in column_details:
            logging.debug(f"{column_detail}")

        logging.debug(f"output labels ({len(labels)}): ")
        for label in labels:
            logging.debug(f"{label}")

    return column_details, labels


def _drop_labels(df, all_labels, labels_to_keep):
    assert type(labels_to_keep) == list, f"labels_to_keep must be list"
    labels_to_drop = list(set(all_labels) - set(labels_to_keep))
    df = df.drop(labels=labels_to_drop, axis=1)
    return df


def _cleanse_data(
    df,
    label_column,
    label_info,
):
    df = df[df[label_column] != 0]
    legal_values = label_info[label_column]
    iw = {num: lab for num, lab in legal_values}
    new_iw = {}
    columns = list(df.columns)
    values = df.values
    label_idx = columns.index(label_column)
    for new_lab, old_lab in enumerate(iw.keys()):
        values[values[:, label_idx] == old_lab, label_idx] = new_lab
        new_iw[new_lab] = iw[old_lab]

    df = pd.DataFrame(values, columns=columns)
    df = df.interpolate()
    pos = df.isnull().stack()[lambda x: x].index.tolist()
    df = df.dropna(axis=0)

    return df, new_iw, pos


def _get_label_info(
    addrs,
):
    data = []
    with open(addrs, mode="r", encoding="utf8") as f:
        for line in f:
            content = [item.strip() for item in line.strip("\n ").split("-")]
            try:
                label_num = int(content[0])
                label_type = content[1]
                label = content[2]
                data += [(label_num, label_type, label)]
            except:
                pass
    label_info = defaultdict(list)
    for num, tip, lab in data:
        label_info[tip] += [(num, lab)]
    return label_info


def _get_data(
    addrs,
):
    raw_data = np.loadtxt(addrs)
    return raw_data


def _get_all_data(
    root,
    labels_to_keep=None,
    num_of_users=4,
    num_of_data_per_user=5,
    number_of_all_columns=250,
    general_frmt=lambda s, f: f"S{s}-ADL{f}.dat",
    drill_frmt=lambda s: f"S{s}-Drill.dat",
    sensor_path=r"sensor_names.txt",
    column_path=r"column_names.txt",
):
    data_columns, label_columns = _get_column_details(
        sensor_path=sensor_path,
        column_detail_path=f"{join(root, column_path)}",
    )
    all_labels = [item[-1] for item in label_columns]
    indices_to_keep, names_indices_to_keep = _columns_to_keep(
        data_columns=data_columns,
        label_columns=label_columns,
    )
    indices_to_remove = _columns_to_remove(
        indices_to_keep=indices_to_keep,
        number_of_columns=number_of_all_columns,
    )
    data = defaultdict(dict)
    for user in range(num_of_users):
        drill_addrs = join(root, drill_frmt(user + 1))
        tmp = _get_data(
            addrs=drill_addrs,
        )
        tmp = _drop_columns(
            data=tmp,
            columns_to_drop=indices_to_remove,
        )
        df = pd.DataFrame(tmp, columns=names_indices_to_keep)
        if labels_to_keep is not None:
            df = _drop_labels(
                df=df,
                all_labels=all_labels,
                labels_to_keep=labels_to_keep,
            )
        data[user]["drill"] = df
        for data_num in range(num_of_data_per_user):
            general_addrs = join(root, general_frmt(user + 1, data_num + 1))
            tmp = _get_data(
                addrs=general_addrs,
            )
            tmp = _drop_columns(
                data=tmp,
                columns_to_drop=indices_to_remove,
            )
            df = pd.DataFrame(tmp, columns=names_indices_to_keep)
            if labels_to_keep is not None:
                df = _drop_labels(
                    df=df,
                    all_labels=all_labels,
                    labels_to_keep=labels_to_keep,
                )
            data[user][data_num] = df
    return data


def _unify(
    root,
    label_column,
    label_path,
    sensor_path,
    column_detail_path,
    num_of_users=4,
):
    """label_column can be one of
    the following items:
    - "HL_Activity"
    - "Locomotion"
    - "ML_Both_Arms"
    - "LL_Left_Arm"
    - "LL_Left_Arm_Object"
    - "LL_Right_Arm"
    - "LL_Right_Arm_Object"
    """
    label_info = _get_label_info(
        addrs=label_path,
    )
    data = _get_all_data(
        root=root,
        labels_to_keep=[label_column],
        num_of_users=num_of_users,
        sensor_path=sensor_path,
        column_path=column_detail_path,
    )
    all_data = []
    total_data = 0
    for user_num in tqdm(range(num_of_users)):
        for dataset in list(range(5)) + ["drill"]:
            df = data[user_num][dataset]
            df, iw, pos = _cleanse_data(
                df=df,
                label_column=label_column,
                label_info=label_info,
            )
            if df.shape[0] > 0:
                all_data += [df.values[:, 1:]]
                total_data += df.values.shape[0]

    # (m, 109)
    unified_data = np.concatenate(all_data, axis=0)

    return unified_data, iw, total_data, df.columns


def load_data(
    root,
    label_column,
    label_path,
    sensor_path,
    column_detail_path,
    frequency=32,
    sampling_time=3,
    hop_size=30,
    seed=1,
    test_val_size=0.15,
):
    unified_data, iw, total_data, columns = _unify(
        root=root,
        label_column=label_column,
        label_path=label_path,
        sensor_path=sensor_path,
        column_detail_path=column_detail_path,
        num_of_users=4,
    )
    x_total, y_total = _make_frames(
        data=unified_data,
        columns=columns[1:],
        frequency=frequency,
        sampling_time=sampling_time,
        hop_size=hop_size,
    )
    (x_train, y_train), (x_tmp, y_tmp) = windower.shuffle_data(
        x=x_total,
        y=y_total,
        seed=seed,
        test_size=test_val_size * 2,
    )
    (x_val, y_val), (x_test, y_test) = windower.shuffle_data(
        x=x_tmp,
        y=y_tmp,
        seed=seed,
        test_size=0.5,
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), iw


def _make_frames(
    data,
    columns,
    frequency=32,
    sampling_time=3,
    hop_size=30,
):
    """The multiplication
    of frequency and sampling_time
    will be used as the length of each window.
    The values provided as the default arguments
    are not exact, but the goal is just the multiplication
    of the mentioned arguments.
    """
    df = windower.make_dataframe(
        data=data,
        columns=columns,
    )
    types = ["float32"] * df.shape[1]
    assert len(types) == len(columns)
    # data type fixer:
    df, _ = windower.make_df_ready(
        df=df,
        cols=columns,
        types=types,
        label_column=None,
        new_label_column_title=None,
    )
    x = df[columns[:-1]]
    y = df[columns[-1:]]
    normalizer = StandardScaler()
    normalized_x = normalizer.fit_transform(X=x)
    normalized_x = pd.DataFrame(data=normalized_x, columns=columns[:-1])
    normalized_x[columns[-1]] = y
    normalized_df = normalized_x
    frame_size = sampling_time * frequency
    frames, labels = windower.frame_preparation(
        df=normalized_df,
        frame_size=frame_size,
        hop_size=hop_size,
        data_columns=columns[:-1],
        label_column=columns[-1],
    )
    labels = labels.astype("int32")
    x_total, y_total = frames, labels

    return x_total, y_total


if __name__ == "__main__":
    root = r"/home/madmedi/Documents/coding/university/capsule/3_har_caps/data_code/1_opportunity/opportunity/dataset"
    data_columns, label_columns = _get_column_details(
        sensor_path=r"./sensor_names.txt",
        column_detail_path=f"{join(root, 'column_names.txt')}",
    )  #
    user_id = 3
    turn = 0
    data = _get_data(
        addrs=join(root, (lambda s, f: f"S{s}-ADL{f}.dat")(user_id + 1, turn + 1))
    )
    logging.debug(f"data: {data.shape}")
    indices_to_keep, names_indices_to_keep = _columns_to_keep(
        data_columns=data_columns,
        label_columns=label_columns,
    )
    logging.debug(f"Columns to keep: {indices_to_keep}")
    logging.debug(f"len: {len(indices_to_keep)}")
    logging.debug("==================================================")
    logging.debug(f"Columns to keep: {names_indices_to_keep}")
    indices_to_remove = _columns_to_remove(
        indices_to_keep=indices_to_keep,
        number_of_columns=data.shape[1],
    )
    logging.debug(f"indices to remove: {indices_to_remove}")
    logging.debug(f"len of indices to remove: {len(indices_to_remove)}")
    data = _drop_columns(
        data=data,
        columns_to_drop=indices_to_remove,
    )
    logging.debug(data.shape)
    assert len(indices_to_keep) == data.shape[1]
    assert len(indices_to_keep) == len(names_indices_to_keep)
    df = pd.DataFrame(data, columns=names_indices_to_keep)
    logging.debug(df.shape)
    logging.debug(f"\n{df.head()}")
    logging.debug("")
    logging.debug("Loco")
    logging.debug(df["Locomotion"].value_counts())
    logging.debug("")
    logging.debug("HL_Activity")
    logging.debug(df["HL_Activity"].value_counts())
    logging.debug("")
    logging.debug("LL_Left_Arm")
    logging.debug(df["LL_Left_Arm"].value_counts())
    logging.debug("")
    logging.debug("LL_Left_Arm_Object")
    logging.debug(df["LL_Left_Arm_Object"].value_counts())
    logging.debug("")
    logging.debug("LL_Right_Arm")
    logging.debug(df["LL_Right_Arm"].value_counts())
    logging.debug("")
    logging.debug("LL_Right_Arm_Object")
    logging.debug(df["LL_Right_Arm_Object"].value_counts())
    logging.debug("")
    logging.debug("ML_Both_Arms")
    logging.debug(df["ML_Both_Arms"].value_counts())
    logging.debug("")
    label_columns = [
        "HL_Activity",
        "Locomotion",
        "ML_Both_Arms",
        "LL_Left_Arm",
        "LL_Left_Arm_Object",
        "LL_Right_Arm",
        "LL_Right_Arm_Object",
    ]
    label_info = _get_label_info(addrs=join(root, r"label_legend.txt"))
    temp = _get_all_data(root=root)
    for label_column in tqdm(label_columns):
        logging.debug("==================================================")
        logging.debug(f"*** label: {label_column}")
        data = _get_all_data(
            root=root,
            labels_to_keep=[label_column],
        )
        for i in range(4):
            for j in range(5):
                assert temp[i][j][label_column].shape[0] > 1_000
                assert all(temp[i][j][label_column] == data[i][j][label_column])
                assert temp[i][j].shape[1] > data[i][j].shape[1]

        for user_num in range(4):
            f = True
            for dataset in list(range(5)) + ["drill"]:
                df = data[user_num][dataset]
                assert df.shape[1] == 110
                assert df.shape[0] > 1_000
                old_df = df
                df, iw, pos = _cleanse_data(
                    df=df,
                    label_column=label_column,
                    label_info=label_info,
                )
                if f:
                    f = False
                    logging.debug(f"*** new_iw: {iw}")
                logging.debug(
                    f"user: {user_num}, data: {dataset}, "
                    f"old df: {old_df.shape}, "
                    f"df: {df.shape}, "
                    f"dropped NaNs: {len(pos)}."
                )
                assert df.shape[1] == old_df.shape[1]
                assert list(old_df.columns) == list(df.columns)
                if not all(old_df[label_column] == 0):
                    if df.shape[0] == 0:
                        msg = f"user_num: {user_num}, dataset: {dataset} is empty."
                        logging.debug(msg)
                    for item in sorted(df[label_column].value_counts().index):
                        assert item in iw.keys()
                else:
                    logging.debug(
                        f">>>>>>>>>> all zeros: user_num: {user_num}, "
                        f"dataset: {dataset}"
                    )

                if any(old_df[label_column] == 0):
                    assert old_df.shape[0] != df.shape[0]
                    if len(df[label_column].value_counts()) > 0:
                        if len(old_df[label_column].value_counts()) != 1 + len(
                            df[label_column].value_counts()
                        ):
                            logging.debug(
                                f"->->->->->->->->->->->->->->->->->->->->->->->->->"
                                f"some labels removed: user: {user_num}, data: {dataset}"
                            )
                else:
                    assert old_df.shape[0] == df.shape[0]
                    assert len(old_df[label_column].value_counts()) == len(
                        df[label_column].value_counts()
                    )
                    logging.debug("** This part")

            logging.debug("**************************************************")
        logging.debug("--------------------------------------------------")

    logging.debug(f"Testing {_unify.__name__}")
    root = r"/home/madmedi/Documents/coding/university/capsule/3_har_caps/data_code/1_opportunity/opportunity/dataset"
    label_column = "HL_Activity"
    label_path = join(root, r"label_legend.txt")
    sensor_path = r"sensor_names.txt"
    column_detail_path = r"column_names.txt"
    unified_data, iw, total_data, columns = _unify(
        root=root,
        label_column=label_column,
        label_path=label_path,
        sensor_path=sensor_path,
        column_detail_path=column_detail_path,
        num_of_users=4,
    )
    assert len(iw.keys()) == 5
    assert unified_data.shape == (total_data, 109)
    assert len(columns) == 110
    logging.debug(f"total_data: {total_data}")

    logging.debug(f"Testing `{_make_frames.__name__}`")
    columns = columns[1:]
    frequency = 32
    sampling_time = 3
    hop_size = 30
    x_total, y_total = _make_frames(
        data=unified_data,
        columns=columns,
        frequency=frequency,
        sampling_time=sampling_time,
        hop_size=hop_size,
    )
    assert type(x_total) == np.ndarray
    assert type(y_total) == np.ndarray
    bs = x_total.shape[0]
    assert x_total.shape[1:] == (frequency * sampling_time, 108)
    assert y_total.shape == (bs,)
    assert len(set(y_total.ravel())) == 5
    assert type(y_total[0]) == np.int32
    logging.debug(f"batch size: {bs}")

    logging.debug(f"Testing `{load_data.__name__}`")
    for label_column in label_columns:
        unified_data, _, _, _ = _unify(
            root=root,
            label_column=label_column,
            label_path=label_path,
            sensor_path=sensor_path,
            column_detail_path=column_detail_path,
            num_of_users=4,
        )
        x_total, _ = _make_frames(
            data=unified_data,
            columns=columns,
            frequency=frequency,
            sampling_time=sampling_time,
            hop_size=hop_size,
        )
        bs = x_total.shape[0]
        (x_train, y_train), (x_val, y_val), (x_test, y_test), iw = load_data(
            root=root,
            label_column=label_column,
            label_path=label_path,
            sensor_path=sensor_path,
            column_detail_path=column_detail_path,
            frequency=frequency,
            sampling_time=sampling_time,
            hop_size=hop_size,
        )
        assert x_train.shape[0] - math.floor(bs * 0.7) <= 1
        assert x_val.shape[0] - math.floor(bs * 0.15) <= 1
        assert x_test.shape[0] - math.floor(bs * 0.15) <= 1
        logging.debug(
            f"label column: {label_column}, labels: {iw}, "
            f"train: {x_train.shape[0]}, "
            f"val: {x_val.shape[0]}, test: {x_test.shape[0]}"
        )

    logging.debug("preprocess.py [Done]")
