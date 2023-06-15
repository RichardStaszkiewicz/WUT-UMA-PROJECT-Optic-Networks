import pandas as pd
import numpy as np
from statistics import stdev
from numpy import average


def split_by_class(data, target):
    class_data = {}

    for i, idx in enumerate(data.index):
        if target[idx] not in class_data.keys():
            class_data[target[idx]] = []
        class_data[target[idx]].append(data.iloc[i].to_dict())

    for key, value in class_data.items():
        class_data[key] = pd.DataFrame(value)

    return class_data


def calculate_avg_for_continuous_attributes(data, attributes):
    avgs = {}
    for attribute in attributes:
        avgs[attribute] = average(data[attribute])
    return avgs


def calculate_stdev_for_continuous_attributes(data, attributes):
    stdevs = {}
    for attribute in attributes:
        stdevs[attribute] = stdev(data[attribute])
    return stdevs


def calculate_probability_for_discrete_attributes(data, attributes):
    probs = {}
    for attribute in attributes:
        count = data[attribute].value_counts().to_dict()
        count_all = sum([x for x in count.values()])
        probs[attribute] = {
            key: count[key] / count_all for key in count.keys()
        }

    return probs


def calculate_probability_norm(x, avg, stdev):
    if stdev == 0:
        if x == avg:
            return 1
        return 0
    exponent = np.exp(-((x - avg) ** 2 / (2 * stdev**2)))
    return exponent / (np.sqrt(2 * np.pi) * stdev)


def create_model(data, target, continuous_attributes, discrete_attributes):
    num_of_sample = len(data)
    class_data = split_by_class(data, target)
    avgs, stdevs = {}, {}
    probs, class_probs = {}, {}
    for class_v in class_data.keys():
        avgs[class_v] = calculate_avg_for_continuous_attributes(
            class_data[class_v], continuous_attributes
        )
        stdevs[class_v] = calculate_stdev_for_continuous_attributes(
            class_data[class_v], continuous_attributes
        )
        probs[class_v] = calculate_probability_for_discrete_attributes(
            class_data[class_v], discrete_attributes
        )
        class_probs[class_v] = len(class_data[class_v]) / num_of_sample
    return avgs, stdevs, probs, class_probs


def calculate_indv_in_class_prob(
    class_avgs, class_stdevs, probs, class_prob, indv
):
    probability = class_prob
    for attribute in class_avgs.keys():
        probability *= calculate_probability_norm(
            indv[attribute], class_avgs[attribute], class_stdevs[attribute]
        )

    for attribute in probs.keys():
        probability *= probs[attribute].get(indv[attribute], 0.0001)

    return probability


def predict(avgs, stdevs, probs, class_probs, indv):
    best_prob = 0
    best_class = None
    for class_id in avgs.keys():
        prob = calculate_indv_in_class_prob(
            avgs[class_id],
            stdevs[class_id],
            probs[class_id],
            class_probs[class_id],
            indv,
        )
        if prob >= best_prob:
            best_prob = prob
            best_class = class_id
    return best_class


def cross_validation(
    train_data,
    train_target,
    bins,
    continuous_attributes,
    discrete_attributes,
    k,
):
    step = len(train_data) // k
    mae = []
    for i in range(k):
        test_data = train_data.iloc[step * i : step * (i + 1)]
        learn_data = pd.concat(
            [
                train_data.iloc[0 : step * i],
                train_data.iloc[step * (i + 1) : -1],
            ]
        )

        test_tar = train_target.iloc[step * i : step * (i + 1)]
        learn_tar = pd.concat(
            [
                train_target.iloc[0 : step * i],
                train_target.iloc[step * (i + 1) : -1],
            ]
        )

        learn_tar_bins = pd.cut(x=learn_tar, bins=bins)

        avgs, stdevs, probs, class_probs = create_model(
            learn_data,
            learn_tar_bins,
            continuous_attributes,
            discrete_attributes,
        )

        absolute_error = 0
        for i, idx in enumerate(test_tar.index):
            prediction = predict(
                avgs, stdevs, probs, class_probs, test_data.iloc[i]
            )
            absolute_error += abs(test_tar[idx] - prediction.mid)
        mae.append(absolute_error / len(test_data))
    return (
        sum(mae) / k,
        create_model(
            train_data,
            pd.cut(x=train_target, bins=bins),
            continuous_attributes,
            discrete_attributes,
        ),
    )


if __name__ == "__main__":
    data = pd.read_csv("./data/data_janos-us-ca.xml.csv")

    train_data = data.iloc[: len(data) // 4]
    test_data = data.iloc[len(data) // 4 :]

    bins = list(range(0, 80, 5))
    mae, (avgs, stdevs, probs, class_probs) = cross_validation(
        train_data,
        train_data["OSNR"],
        bins,
        ["hop_len", "no_of_hops", "avg_hop_loss"],
        ["transponder_modulation", "transponder_bitrate"],
        6,
    )

    print("MAE cross validation:", mae)

    abslute_error = 0
    for i, idx in enumerate(test_data.index):
        prediction = predict(
            avgs, stdevs, probs, class_probs, test_data.iloc[i]
        )
        abslute_error += abs(test_data["OSNR"][idx] - prediction.mid)

    print("MAE without cross validation:", abslute_error / len(test_data))
