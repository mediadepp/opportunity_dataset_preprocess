# IHSN
__author__ = "Mad Medi"
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import time
from scipy import stats
from utils import logger, timer, logger_path


@timer
def dataframe_stats(df):
    logger(df.head())
    logger(f"dataframe samples: {len(data)} \n")
    logger(f"df shape: {df.shape} \n")
    logger(f"info: {df.info()} \n")
    logger(f"isnull: {df.isnull().sum()} \n")
    logger(f"counts: {df['activity'].value_counts()} \n")
    per = df["activity"].value_counts() / len(data) * 100
    logger(f"percentage: {per} \n")


@timer
def drop_columns(df, cols):
    df = df.drop(cols, axis=1).copy()
    return df


@timer
def get_labels(df, col):
    res = df[col].value_counts().index
    res = list(res)
    return res


@timer
def load_data_wsdm(path):
    lines = []
    with open(path, mode="r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            try:
                items = line.strip().split(",")
                items = [item.strip(" ;") for item in items]
                items = [item for item in items if item != ""]
                assert len(items) == 6
                lines += [items]
            except Exception as e:
                logger(e, f"line number: {idx}, len: {len(items)}, items: {items}")
    return lines


@timer
def make_dataframe(data, columns):
    """Makes dataframe out of the input list.

    :param data: A list of lists
    :type data: List[List[str]]
    """
    data = pd.DataFrame(data=data, columns=columns)
    return data


@timer
def make_df_ready(df, cols, types, label_column=None, new_label_column_title=None):
    for col, typ in zip(cols, types):
        df[col] = df[col].astype(typ)

    classes = None
    if label_column is not None:
        le = LabelEncoder()
        df[new_label_column_title] = le.fit_transform(df[label_column])
        classes = le.classes_
    return df, classes


@timer
def plot_activity(df, label, horizon, cols):
    num = len(cols)
    fig, axes = plt.subplots(
        nrows=num,
        figsize=(num * 5, 7),
        sharex=True,
    )
    for idx, col in enumerate(cols):
        plot_axis(
            axis=axes[idx],
            x=df[horizon],
            y=df[col],
            title=f"{col}-axis",
        )
    plt.subplots_adjust(hspace=0.2)
    plt.suptitle(label)
    plt.subplots_adjust(top=0.9)
    plt.show()


@timer
def plot_axis(axis, x, y, title):
    axis.plot(x, y, "g")
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([np.min(y) - np.std(y), np.max(y) + np.std(y)])
    axis.set_xlim([np.min(x), np.max(x)])
    axis.grid(True)


@timer
def frame_preparation(
    df,
    frame_size,
    hop_size,
    data_columns,
    label_column,
):
    frames = []
    labels = []
    inp = df[data_columns].values
    out = df[label_column].values
    for idx in range(0, len(df) - frame_size, hop_size):
        x = inp[idx : (idx + frame_size), :]
        y = out[idx : (idx + frame_size)]
        frames += [x]
        labels += [stats.mode(y)[0]]
    frames = np.array(frames)  # (bs, #frames, #features)
    labels = np.array(labels)  # (bs, )
    return frames, labels


@timer
def shuffle_data(x, y, seed=1, test_size=0.25):
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=y,
        )
    except:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=None
        )

    return (x_train, y_train), (x_test, y_test)


@timer
def make_model(name, ts, num_features, num_cls):
    model = tf.keras.models.Sequential(name=name)
    model.add(
        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=5,
            padding="valid",
            activation="relu",
            input_shape=(ts, num_features),
        )
    )  # (bs, 78, 32)
    model.add(
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            padding="valid",
            activation="relu",
            input_shape=(ts, num_features),
        )
    )  # (bs, 76, 32)
    model.add(tf.keras.layers.MaxPool1D(2))  # (bs, 38, 32)
    model.add(tf.keras.layers.Flatten())  # (bs, 38*32)
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(num_cls, activation="linear"))
    return model


@timer
def plot_conf(y_true, y_pred, cls_names):
    y_pred = tf.argmax(y_pred, axis=1)
    conf = confusion_matrix(y_true, y_pred)
    fig, ax = plot_confusion_matrix(
        conf_mat=conf,
        class_names=cls_names,
        show_normed=True,
        figsize=(10, 10),
    )
    return fig, ax


if __name__ == "__main__":
    if os.path.exists(logger_path):
        os.remove(logger_path)
    path = r"./dataset.txt"
    data = load_data_wsdm(path)
    columns = ["user", "activity", "time", "x", "y", "z"]
    df = make_dataframe(
        data=data,
        columns=columns,
    )
    dataframe_stats(df)
    cols = [
        "x",
        "y",
        "z",
    ]
    types = [
        "float32",
        "float32",
        "float32",
    ]
    df, labs = make_df_ready(
        df=df,
        cols=cols,
        types=types,
        label_column="activity",
        new_label_column_title="label",
    )
    labels = get_labels(
        df=df,
        col="activity",
    )
    logger("========================= labels =========================")
    logger(labels)
    logger(labs)
    hz = 20
    frames = 10 * hz
    if False:
        for activity in labels:
            d = df[df["activity"] == activity][:frames]
            plot_activity(
                df=d,
                label=activity,
                horizon="time",
                cols=["x", "y", "z"],
            )
        df = drop_columns(df=df, cols=["time", "user"])
    logger("\n\n", "\n\n", f"{df.head()}")
    logger("\n\n", "\n\n", f"{df['activity'].value_counts()}")
    x = df[["x", "y", "z"]]
    y = df[["label"]]
    normalizer = StandardScaler()
    normalized_x = normalizer.fit_transform(X=x)  # the output is a numpy array
    normalized_x = pd.DataFrame(data=normalized_x, columns=["x", "y", "z"])
    normalized_x["label"] = y
    normalized_ds = normalized_x
    logger("normalized dataset: ")
    logger(normalized_ds.head())
    logger(f"\n\n {normalized_ds['label'].value_counts()} \n\n")
    assert normalized_ds["x"].mean() <= 0.01
    assert normalized_ds["y"].mean() <= 0.01
    assert normalized_ds["z"].mean() <= 0.01
    frequency = 20  # hz
    sampling_time = 4  # seconds (s)
    frame_size = sampling_time * frequency
    hop_size = int((sampling_time / 2) * frequency)
    num_features = 3  # x, y, z
    data_columns = ["x", "y", "z"]
    label_column = "label"
    frames, labels = frame_preparation(
        df=normalized_ds,
        frame_size=frame_size,
        hop_size=hop_size,
        data_columns=data_columns,
        label_column=label_column,
    )
    bs = frames.shape[0]
    assert frames.shape == (bs, frame_size, num_features)
    assert labels.shape == (bs,)
    x_total, y_total = frames, labels
    logger(f"{x_total.shape} \t {y_total.shape}")
    (x_train, y_train), (x_test, y_test) = shuffle_data(
        x=x_total,
        y=y_total,
        seed=1,
    )
    logger(f"x_train: {x_train.shape}")
    logger(f"y_train: {y_train.shape}")
    logger(f"x_test: {x_test.shape}")
    logger(f"y_test: {y_test.shape}")
    logger()
    num_cls = len(set(y_total))
    model = make_model(
        "my_model", ts=frame_size, num_features=num_features, num_cls=num_cls
    )
    logger(model.summary())
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), "accuracy"],
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=50,
        validation_split=0.1,
    )
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.legend()
    plt.show()
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["loss"], label="loss")
    plt.legend()
    plt.show()
    y_pred = model(x_test)
    a, b = plot_conf(y_test, y_pred, labs)
    plt.show()
    logger("Done")
