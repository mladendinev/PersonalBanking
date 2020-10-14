import sys

import numpy
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from datetime import datetime

class FileReader:
    def convert_xls_to_xlsx(self, input_file, output_file):
        df = pd.read_html(input_file, header=0)
        df[0].to_excel("../data/" + output_file, index=0)

    def last_file(self, base_path):
        files = os.listdir(base_path)
        paths = [os.path.join(base_path, basename) for basename in files]
        return max(paths, key=os.path.getctime)

    def read_last_sheet(self):
        last_inserted_file = self.last_file('../data')
        filename = os.path.splitext(last_inserted_file.split('/')[-1])

        parser = lambda date: datetime.strptime(date, '%d.%m.%Y %H:%M:%S')
        df = pd.read_excel(last_inserted_file, parse_dates=['Дата на плащане'], date_parser=parser)
        df.sort_values(by=['Дата на плащане'], inplace=True)
        return df, filename[0]


class Report:
    def __init__(self, banking_data, note=None):
        self.banking_data = banking_data
        self.note = note

    def count_transactions_per_week(self, axs):
        axs.set_title("Number of transactions per week")
        incoming_records = self.banking_data.loc[self.banking_data['Тип'] == "ДТ"]
        axs.hist(incoming_records['седмица'], histtype="bar", bins=range(1, 6), color='cornflowerblue')

    def autolabel(self, rects, axs):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axs.annotate('{}'.format(height.astype(int)),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='center')

    def types_of_transactions_per_week(self, axs):
        df3 = (
            self.banking_data
                .groupby(by=['седмица', 'Тип'], dropna=False)
                .sum()
                .reset_index()
        )

        labels = ["w1", "w2", "w3", "w4", "w5"]
        income = numpy.zeros(len(labels))
        expenses = numpy.zeros(len(labels))
        print(df3.values)
        for week, transaction_type, amount in df3.values:
            if transaction_type == "ДТ":
                expenses[week] = amount
            else:
                income[week] = amount

        # figure properties
        x = numpy.arange(len(labels))
        width = 0.35

        bar1 = axs.bar(x - width / 2, income, width, label='Incoming')
        bar2 = axs.bar(x + width / 2, expenses, width, label='Outgoing')
        axs.legend(loc="upper left")
        axs.set_title("Type of transactions per week")
        axs.set_xticks(x)
        axs.set_xticklabels(labels)

        self.autolabel(bar1, axs)
        self.autolabel(bar2, axs)

    def expenses_savings_pie(self, expenses, savings, axs):
        labels = 'Expenses', 'Savings'
        axs.pie([expenses, savings], autopct='%1.1f%%', labels=labels)
        axs.axis('equal')
        axs.set_title("Pie Chart")

    def income_expenses_savings_bar_chart(self, expenses, income, savings, axs):
        axs.set_title("Income Expenses Savings")
        labels = ['Expenses', 'Income', 'Savings']
        values = [expenses, income, savings]
        axs.bar(labels, values, color=['cornflowerblue'])
        for i in range(len(values)):
            axs.annotate(str(values[i]), xy=(labels[i], values[i]), ha='center', va='center')

    def income_expenses_savings(self):
        expenses = self.banking_data.loc[self.banking_data['Тип'] == 'ДТ', 'Сума във валута на сметката'].sum()
        income = self.banking_data.loc[self.banking_data['Тип'] == 'КТ', 'Сума във валута на сметката'].sum()
        savings = income - expenses

        return expenses, income, 0 if savings < 0 else savings

    def visualise_report(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))

        # week of the month not in the year
        self.banking_data['седмица'] = (self.banking_data["Дата на плащане"].dt.day - 1) // 7 + 1

        expenses, income, savings = self.income_expenses_savings()
        self.expenses_savings_pie(expenses, savings, ax1)
        self.income_expenses_savings_bar_chart(expenses, income, savings, ax2)
        self.count_transactions_per_week(ax3)
        self.types_of_transactions_per_week(ax4)
        plt.tight_layout()
        plt.show()

        output_figure_file = note + ".png"
        if os.path.isfile(output_figure_file):
            print("already existing file")
            sys.exit(0)
        plt.savefig(output_figure_file)


class MonthlyReport(Report):
    def __init__(self, excel_data, note=None):
        self.monthly_data = excel_data[excel_data["Дата на плащане"].dt.month == datetime.today().month]
        super().__init__(self.monthly_data, note)


class GeneralReport(Report):
    def __init__(self, excel_data):
        super().__init__(excel_data)


if __name__ == "__main__":
    data, note = FileReader().read_last_sheet()
    monthly_report = MonthlyReport(data)
    monthly_report.visualise_report()

    # monthly_report = GeneralReport(data)
    # monthly_report.visualise_report()
