import os

import pandas as pd


def convert_xls_to_xlsx(input_file, output_file):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "../data/" + input_file)
    df = pd.read_excel(path, header=0)
    df.drop(df.columns.values[0], axis=1, inplace=True)
    df.to_excel("../data/" + output_file, index=0)


convert_xls_to_xlsx("report_first_half_october_copy.xlsx", "report_first_half_october.xlsx")
