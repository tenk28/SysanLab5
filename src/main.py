import sys
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtGui

from interface import Ui_MainWindow
from delphi import Delphi


class UI(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.output_text.setFont(QtGui.QFont("Consolas", 10))
        self.output_text.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
        self.choose_input_button.clicked.connect(self.choose_input_file)
        self.execute_button.clicked.connect(self.execute)
        self.input_filename.setText('km_1.xlsx')

    def choose_input_file(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open data file', '.', 'Data file (*.xlsx)')[0]
        self.input_filename.setText(filename)

    def execute(self):
        input_file = self.input_filename.text()
        delphi = Delphi(input_file)
        quality_result_table = delphi.quality_result_table
        quality_result_table.index = delphi.events.tolist()
        quality_result_table.columns = delphi.indicators.tolist() + ['W']
        for row in quality_result_table.index:
            for column in quality_result_table.columns[:-1]:
                quality_result_table.loc[row, column] = delphi.marks[quality_result_table.loc[row, column]]
        quantity_result_table = delphi.quantity_result_table
        quantity_result_table.index = delphi.events.tolist()
        quantity_result_table.columns = delphi.indicators.tolist() + ['W']
        output = 'Результати аналізу експертного оцінювання ММД\n'
        for event_name, event in zip(delphi.events, delphi.event_results):
            output += event_name + '\n'
            event.index = ['T', '%', 'M', 'S', 'Q', 's']
            event.columns = delphi.indicators[event.columns]
            output += event.to_string(index_names='') + '\n'
        output += 'Узгоджені кількісні оцінки експертного опитування\n' \
                  + quantity_result_table.to_string(index_names='') + '\n' + \
                  'Узгоджені кількісні оцінки експертного опитування\n' + quality_result_table.to_string(index_names='')
        self.output_text.setText(output)

    def _format_result_table(self, result_table):
        ...


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = UI()
    MainWindow.show()
    sys.exit(app.exec())