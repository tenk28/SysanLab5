import pandas as pd


class Preprocessing:
    def __init__(self, base_file, estimates_file):
        """
        :param str base_file: шлях до файлу з базовими коефіцієнтами
        :param (str, str) estimates_file: шлях до файлу з експертними оцінками
        """
        self.base_file = base_file
        self.estimates_file, self.event = estimates_file

        self.weights = None  # Ваги показників
        self.k = None  # Радіус довірчого інтервалу
        self.s = None  # Граничний рівень узгодженості
        self.r = None  # Коефіцієнт врахування ваги показників впевненості експертів
        self.competence = None  # Значення компетентності експертів

        self.index = None  # Series показників з розшифровкою
        self.estimates = None  # Список з DataFrame оцінок експертів для кожного з критеріїв

        self.base_processing()
        self.estimates_processing()

    def base_processing(self):
        """Коректне завантаження даних з файлу з базовими коефіцієнтами."""
        self.weights = pd.read_excel(self.base_file, 'Weights', header=None, index_col=0).squeeze()
        coefficients = pd.read_excel(self.base_file, 'KSR', header=None, index_col=0).squeeze()
        self.k = coefficients['K']
        self.s = coefficients['S']
        # self.r = coefficients['R']
        self.r = 0.55  # встановлено як мінімальний радіус, за якого всі коефіцієнти узгодженості більші за self.s
        self.competence = pd.read_excel(self.base_file, 'Competence', header=None, index_col=0).squeeze()
        self.index = pd.read_excel(self.base_file, 'Indicators', header=None, index_col=0).squeeze()

    def estimates_processing(self):
        """Коректне завантаження даних з файлу з  експертними оцінками."""
        self.estimates = [pd.read_excel(self.estimates_file, self.event+i, header=None, index_col=0)
                          for i in self.index.index]

