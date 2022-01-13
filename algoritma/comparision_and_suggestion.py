from tensorflow.tools.docs.doc_controls import T
from yfinance import ticker
from create_data import ticker_list, ebit_list, ev_list, ebit_ev_list, predictions, roc_list, stock_last_value_prediction_value_difference
import random


day = 5
length = len(ticker_list)

data_per_ticker = {}
for i in range(length):
    temp_dict = {}
    temp_dict["ev"] = ev_list[ticker_list[i]]
    temp_dict["ebit/ev"] = ebit_ev_list[ticker_list[i]]
    # temp_dict["ev"]
    # temp_dict["ebit/ev"]
    temp_dict["prediction"] = predictions[ticker_list[i]]
    temp_dict["roc"] = roc_list[ticker_list[i]]
    temp_dict["stock_change_percentage"] = stock_last_value_prediction_value_difference[ticker_list[i]]
    data_per_ticker[ticker_list[i]] = temp_dict

# for key, value in data_per_ticker.items():
#     print("Ticker:{}, Values:{}".format(key, value))


def make_suggestion(data):
    points = {}
    for key, value in data.items():
        points[key] = value["ev"] * 0.01 + value["ebit/ev"] * 0.02
        points[key] += value["stock_change_percentage"] + value["roc"] * 0.03
    points_items = points.items()
    points_sorted = sorted(points_items)
    return points_sorted


suggestions = make_suggestion(data_per_ticker)
print("For {} days, the best stock choice is:{}, prediction for the {} stock is {}".format(
    day, suggestions[-1][0], suggestions[-1][0], predictions[suggestions[-1][0]][0]))
