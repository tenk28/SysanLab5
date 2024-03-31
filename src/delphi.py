import pandas as pd
import numpy as np
from scipy.optimize import minimize

from load import Preprocessing

MILLER_SCALE = [1, 2, 3, 4, 5, 6, 7]  # оцінки у шкалі Міллера
NU = [2, 4, 6, 8, 10, 12, 14]  # номери колон для впевненості експертів у оцінках
MU = [1, 3, 5, 7, 9, 11, 13]  # номери колон для оцінок експертів


class Delphi:
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.marks = {1: 'Надзвичайно низький', 2: 'Дуже низький', 3: 'Низький', 4: 'Середній', 5: 'Високий',
                      6: 'Дуже високий', 7: 'Надзвичайно високий'}
        self.events = pd.read_excel(self.input_filename, 'Events', index_col=0, header=None).squeeze()
        self.indicators = pd.read_excel(self.input_filename, 'Indicators', header=None, index_col=0).squeeze()
        self.quantity_result_table, self.quality_result_table, self.event_results = self.get_results()

    def get_results(self):
        """Отримати результати методу Делфі у вигляді таблиць."""
        weights = []  # зважені оцінки
        assessment_results = []  # таблиці зважених результатів
        indicator = []
        marks_value = []
        estimates_value = []
        for event in self.events.index:
            ed = EventDelphi(self.input_filename, (self.input_filename, event), self.input_filename)
            ar, w = ed.get_result()
            marks_value.append(ar.loc['Оцінка'].tolist())
            estimates_value.append(ed.agreed_expert_assessment)
            indicator.append(ar.shape[1])
            weights.append(w)
            assessment_results.append(ar)
        indicator = [f'I{i+1}' for i in np.arange(max(indicator))]
        results_quality = pd.DataFrame(marks_value, index=self.events.index, columns=indicator)
        results_quantity = pd.DataFrame(estimates_value, index=self.events.index, columns=indicator)
        results_quality['W'] = weights
        results_quantity['W'] = weights

        return results_quantity, results_quality, assessment_results


class EventDelphi(Preprocessing):
    def __init__(self, base_file, estimates_file, save_file):
        """
        :param str base_file: шлях до файлу з базовими коефіцієнтами
        :param (str, str) estimates_file: шлях до файлу з експертними оцінками
        :param pd.ExcelWriter save_file: шлях до файлу для збереження
        """
        super(EventDelphi, self).__init__(base_file, estimates_file)
        self.save_file = save_file
        self.median_expert = None

        self.d = self.get_interval_evaluations()  # інтервальні оцінки
        self.m = self.get_interval_average_score()  # середні інтервальні оцінки
        self.q = self.get_model_expert_assessment()  # модельні експертні оцінки
        self.sf = self.get_scaling_factor()  # коефіцієнт масштабування до інтервалу [0; 1]
        self.gd = self.get_interval_gaussian_density()  # інтервальна гаусівська щільність
        self.qf = self.get_quality_function()  # функціонал якості експертів
        self.dm = self.get_distance_matrix()  # матриця відстаней між експертними оцінками
        self.dv = self.get_distance_vector()  # вектор відстаней експертних оцінок
        self.median = self.get_median()  # медіана інтервальної оцінки
        self.cim = self.get_confidence_interval_members()  # експерти з оцінками в довірчому інтервалі
        self.consistency_indicator = self.get_consistency_indicator()  # показник узгодженості
        self.agreed_expert_mark = self.get_agreed_expert_mark()  # узгоджені експертні оцінки
        self.agreed_expert_assessment = self.get_agreed_expert_assessment()  # узгоджені експертні оцінки
        print()

    def _re_init(self):
        self.d = self.get_interval_evaluations()  # інтервальні оцінки
        self.m = self.get_interval_average_score()  # середні інтервальні оцінки
        self.q = self.get_model_expert_assessment()  # модельні експертні оцінки
        self.sf = self.get_scaling_factor()  # коефіцієнт масштабування до інтервалу [0; 1]
        self.gd = self.get_interval_gaussian_density()  # інтервальна гаусівська щільність
        self.qf = self.get_quality_function()  # функціонал якості експертів
        self.dm = self.get_distance_matrix()  # матриця відстаней між експертними оцінками
        self.dv = self.get_distance_vector()  # вектор відстаней експертних оцінок
        self.median = self.get_median()  # медіана інтервальної оцінки
        self.cim = self.get_confidence_interval_members()  # експерти з оцінками в довірчому інтервалі
        self.consistency_indicator = self.get_consistency_indicator()  # показник узгодженості

    def get_level_thresholds(self, default=True):
        """
        Обчислення порогових значень шкали вимірювань.
        :return list: список списків порогових значень шкали вимірювань для шкали Міллера
        """
        estimates = self.estimates
        p = len(self.index)

        lt = []
        for i in np.arange(p):
            if default:
                x_s = [0.07, 0.21, 0.36, 0.5, 0.64, 0.79, 0.93]
            else:
                nu_max = estimates[i][NU].max().max()
                nu_min = estimates[i][NU].min().min()
                nu_step = (nu_max-nu_min) / 6
                x_s = [nu_min+nu_step*ms for ms in np.arange(7)]
            lt.append(pd.Series(x_s, MILLER_SCALE))
        return lt

    def get_interval_evaluations(self):
        """
        Обчислення інтервальних оцінок для кожного з показників.
        :return list: список кортежів з нижньою та верхньою інтервальною оцінкою відповідно
        """
        estimates = self.estimates
        p = len(self.index)
        k = self.k

        ie = []
        for i in np.arange(p):
            mu = estimates[i][MU].set_axis(MILLER_SCALE, axis='columns')
            nu = estimates[i][NU].set_axis(MILLER_SCALE, axis='columns')
            interval_evaluations_low = (mu - mu*(1-nu)*k).abs().clip(lower=0)  # Нижні межі інтервалів
            interval_evaluations_high = (mu + mu*(1-nu)*k).abs().clip(upper=1)  # Верхні межі інтервалів
            ie.append((interval_evaluations_low, interval_evaluations_high))
        return ie

    def get_interval_average_score(self):
        """
        Обчислення середніх інтервальних оцінок для кожного з показників.
        :return list: список кортежів з нижньою та верхньою середньою інтервальною оцінкою відповідно
        """
        k = len(self.competence)  # Кількість експертів
        p = len(self.index)
        d = self.d

        ias = []
        for i in np.arange(p):
            ias_low = d[i][0].sum('index') / k  # Нижні межі інтервалів
            ias_high = d[i][1].sum('index') / k  # Верхні межі інтервалів
            ias.append((ias_low, ias_high))
        return ias

    def get_model_expert_assessment(self):
        """
        Обчислення модельних експертних оцінок для кожного з показників.
        :return list: список кортежів з нижньою та верхньою модельною експертною оцінкою відповідно
        """
        p = len(self.index)
        d = self.d
        m = self.m

        mea = []
        for i in np.arange(p):
            low_idx = d[i][0].sub(m[i][0], 'columns').abs().idxmin('index')
            # low_idx = self.estimates[i][MU].sub(m[i][0], 'columns').abs().idxmin('index')
            mea_low = pd.Series([d[i][0].loc[low_idx[idx], idx] for idx in MILLER_SCALE],
                                MILLER_SCALE)
            high_idx = d[i][1].sub(m[i][1], 'columns').abs().idxmin('index')
            # low_idx = self.estimates[i][MU].sub(m[i][1], 'columns').abs().idxmin('index')
            mea_high = pd.Series([d[i][1].loc[high_idx[idx], idx] for idx in MILLER_SCALE],
                                 MILLER_SCALE)
            mea.append((mea_low, mea_high))
        return mea

    def get_scaling_factor(self):
        """
        Знаходимо коефіцієнт масштабування до інтервалу [0; 1]
        :return list: список кортежів з двох чисел.
        """
        p = len(self.index)
        lt = self.get_level_thresholds()
        q = self.q

        sf = []
        for i in np.arange(p):
            # Розрахуємо довжини інтервалів
            il = lt[i][1:7].reset_index(drop=True) - lt[i][0:6].reset_index(drop=True)
            # Знайдемо площу під кривою мінімуму
            q_i = (q[i][0][:-1].reset_index(drop=True) + q[i][0][1:].reset_index(drop=True)) / 2
            s_low = (il * q_i).sum()
            # Знайдемо площу під кривою максимуму
            q_i = (q[i][1][:-1].reset_index(drop=True) + q[i][1][1:].reset_index(drop=True)) / 2
            s_high = (il * q_i).sum()
            # Збережемо коефіцієнти масштабування
            sf.append((1/s_low, 1/s_high))
        return sf

    def get_interval_gaussian_density(self):
        """
        Знаходимо інтервальну гаусівську щільність.
        :return list: список кортежів з двох чисел
        """
        p = len(self.index)
        lt = self.get_level_thresholds()
        sf = self.sf
        q = self.q

        def minimized_function(k, x_s, _q_, md):
            """
            Обчислює значення функції.
            :param float k: коефіцієнт масштабування
            :param pd.Series x_s: порогові значення шкали вимірювань
            :param pd.Series _q_: модельні експертні оцінки
            :param float md: математичне сподівання та дисперсія гаусівського розподілу
            :return float: значення функції
            """
            m, d = md
            gaussian_density = calculate_gaussian_density(k, x_s, m, d)
            result = (gaussian_density - _q_).abs().sum()
            return result

        def calculate_gaussian_density(k, x_s, m, d):
            """
            Обчислює значення гаусівської щільності для знайдених m та d.
            :param float k: коефіцієнт масштабування
            :param pd.Series x_s: порогові значення шкали вимірювань
            :param float m: математичне сподівання гаусівського розподілу
            :param float d: дисперсія гаусівського розподілу
            :return float: значення гаусівської щільності
            """
            return 1 / (k * np.sqrt(2*np.pi*d)) * np.exp(-(x_s-m)**2/(2*d))

        gd = []
        for i in np.arange(p):
            median, dispersion = minimize(lambda _md_: minimized_function(sf[i][0], lt[i], q[i][0], _md_),
                                          np.array([0.5, 0.5]),  method='SLSQP', bounds=((1e-6, 1), (1e-6, 1))).x
            gd_low = calculate_gaussian_density(sf[i][0], lt[i], median, dispersion)
            median, dispersion = minimize(lambda _md_: minimized_function(sf[i][1], lt[i], q[i][1], _md_),
                                          np.array([0.5, 0.5]), method='SLSQP', bounds=((1e-6, 1), (1e-6, 1))).x
            gd_high = calculate_gaussian_density(sf[i][1], lt[i], median, dispersion)
            gd.append((gd_low, gd_high))
        return gd

    @staticmethod
    def metric(a, b):
        """
        Пошук інтервальної метрики.
        :param (pd.DataFrame, pd.DataFrame) || (pd.Series, pd.Series)  a: кортеж з двох DataFrame
        :param (pd.DataFrame, pd.DataFrame) || (pd.Series, pd.Series) b: кортеж з двох DataFrame
        :return pd.DataFrame: значення інтервальної метрики
        """
        ro = pd.concat([(a[0]-b[0]).abs(), (a[1]-b[1]).abs()]).groupby(level=0).max()
        return ro

    def get_quality_function(self):
        """
        Обчислення функціоналу якості експертів.
        :return list: список pd.Series
        """
        p = len(self.index)
        d = self.d
        gd = self.gd
        competence = self.competence

        qf = []
        for i in np.arange(p):
            qf_i = []
            for j in competence.index:
                d_j = (d[i][0].loc[j], d[i][1].loc[j])
                gauss = (gd[i][0], gd[i][1])
                ro = self.metric(d_j, gauss).sum() / 7
                w = (1-ro)*competence[j]
                qf_i.append(w)
            qf.append(pd.Series(qf_i, index=competence.index))
        return qf

    def get_distance_matrix(self):
        """
        Обраховує матрицю відстаней між всіх експертних оцінок.
        :return list: список pd.DataFrame
        """
        p = len(self.index)
        experts = self.competence.index
        d = self.d

        dm = []
        for i in np.arange(p):
            dm_i = pd.DataFrame(index=experts, columns=experts, dtype=float)
            for row in experts:
                for column in experts:
                    first = (d[i][0].loc[row], d[i][1].loc[row])
                    second = (d[i][0].loc[column], d[i][1].loc[column])
                    dm_i.loc[row, column] = self.metric(first, second).sum() / 7
            dm.append(dm_i)
        return dm

    def get_distance_vector(self):
        """
        Створює вектор відстаней для експертних оцінок.
        :return list: список pd.Series
        """
        dm = self.dm

        dv = [dm_i.sum() for dm_i in dm]
        return dv

    def get_median(self):
        """
        Пошук медіани інтервальної оцінки | медіани кластеру.
        :return:
        """
        p = len(self.index)
        dv = self.dv
        d = self.d

        m = []
        self.median_expert = []
        for i in np.arange(p):
            m_idx = dv[i].idxmin()
            self.median_expert.append(m_idx)
            median = (d[i][0].loc[m_idx], d[i][1].loc[m_idx])
            m.append(median)
        return m

    def get_confidence_interval_members(self):
        """
        Отримати перелік експертів, чиї оцінки належать до довірчого інтервалу.
        :return list: список pd.Series
        """
        p = len(self.index)
        d = self.d
        gd = self.gd
        m = self.median
        # m = self.m
        c = self.competence
        r = self.r

        def distance(q, median, gauss, competence):
            """
            Визначення відстані між q та median.
            :param (pd.Series, pd.Series) q: інтервальні оцінки
            :param (pd.Series, pd.Series)  median: медіана інтервальних оцінок
            :param (pd.Series, pd.Series) gauss: гаусівська щільність
            :param float competence: компетентність експертів
            :return:
            """
            w = (1-self.metric(q, gauss).sum()/7) * competence
            dist = self.metric(q, median).sum()/7 * (2-w)
            return dist
        cim = []
        # r_t = 0  # новий радіус довірчого інтервалу (непотрібно, оскільки тур лише один)
        for i in np.arange(p):
            cim_i = {}
            for e in c.index:
                _distance_ = distance((d[i][0].loc[e], d[i][1].loc[e]), (m[i][0], m[i][1]), (gd[i][0], gd[i][1]), c[e])
                if _distance_ <= r:
                    cim_i[e] = _distance_
            cim_i = pd.Series(cim_i)
            # r_t = max(r_t, ci_i.max())  # новий радіус довірчого інтервалу (непотрібно, оскільки тур лише один)
            cim.append(cim_i)
        return cim

    def get_consistency_indicator(self):
        """
        Обчислення показника узгодженості.
        :return list: список чисел з крапкою
        """
        e_number = self.competence.size
        cim_numbers = [cim.size for cim in self.cim]

        ci = [number / e_number for number in cim_numbers]
        return ci

    def get_agreed_expert_mark(self):
        """
        Знаходимо узгоджені оцінки експертів.
        :return list: список цілих чисел
        """
        p = len(self.index)
        median_gap = [(median[1]-median[0]).abs() for median in self.median]
        q = [(m[0]+m[1])/2 for m in self.median]

        aem = []
        for i in np.arange(p):
            q_max = q[i].max()
            q_max = [idx for idx in q[i].index if q[i][idx] == q_max]
            median_min = median_gap[i].min()
            median_min = [idx for idx in median_gap[i].index if median_gap[i][idx] == median_min]
            s = [idx for idx in q_max if idx in median_min]
            # Обираємо найменшу з перетину АБО середнє арифметичне двох найменших
            s = s[0] if s else int((q_max[0]+median_min[0])/2)
            aem.append(s)
        return aem

    def get_agreed_expert_assessment(self):
        """
        Знаходимо узгоджені оцінки експертів.
        :return list: список чисел з крапкою
        """
        q = [(m[0] + m[1]) / 2 for m in self.median]
        mark = self.agreed_expert_mark

        aea = [_q_.loc[_mark_] for (_q_, _mark_) in [*zip(q, mark)]]
        return aea

    def weighted_assessment(self):
        """
        Обчислюємо зважену експертну оцінку.
        :return float: зважена експертна оцінка
        """
        w = self.weights
        assessment = pd.Series(self.agreed_expert_assessment, index=w.index)

        return (w*assessment).mean()

    def get_result(self):
        """
        Формуємо таблицю результатів аналізу оцінок експертів.
        :return pd.DataFrame, float: таблиця результатів аналізу оцінок експертів, зважена оцінка експертів
        """
        header = self.index.index
        index = ['Експертів', '%', 'Mid', 'S', 'Q', 'Оцінка']
        experts = [cim.size for cim in self.cim]
        percent = [ci*100 for ci in self.consistency_indicator]
        median = [f'Експерт №{e[1:]}' for e in self.median_expert]
        s = [ci for ci in self.consistency_indicator]
        q = self.agreed_expert_assessment
        mark = self.agreed_expert_mark
        analysis_table = pd.DataFrame([experts, percent, median, s, q, mark], columns=header, index=index)
        return analysis_table, self.weighted_assessment()


# def calculate_confidence_interval_radius():
#     """Допоміжний скрипт для виявлення 'правильного' радіусу довірчого інтервалу."""
#     base = 'БазовіКоефіцієнти.xlsx'
#     estimates = ('ВпровадженняДокументообігуТаАнтивірусногоЗахисту.xlsx', 'ВпровадженняКонцепціїНІКІ.xlsx',
#                  'ВпровадженняХмарнихТехнологій.xlsx', 'РозвитокСтандартів4GТа5G.xlsx')
#     s_prior = 0.65
#     r = 0.3
#     r_step = 0.01
#     s = 0
#     while s < s_prior:
#         s = []
#         for estimate in estimates:
#             d = EventDelphi(base, estimate)
#             d.r = r
#             d._re_init()
#             s += d.get_consistency_indicator()
#         s = min(s)
#         r += r_step
#     return r
