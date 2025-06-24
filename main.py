import sys
import os

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox, QTableWidget,
                             QTableWidgetItem, QInputDialog, QDialog, QLineEdit, QGridLayout, QStackedWidget, QComboBox,
                             QSizePolicy, QSpinBox)
from PyQt5.QtCore import Qt
import pandas as pd
from FCM import setup_fcm
import openpyxl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from FCM import run_impulse_simulation
from FCM import get_global_data
from FCM import alg_hebba
import csv
import re

matrix = []
column = []
app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("EcoFCM")
window.setGeometry(100, 100, 800, 800)

import sys
import os
from PyQt5.QtCore import QFile, QTextStream, QIODevice

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

style_path = resource_path("style.qss")
icon_path = resource_path("iconka.png")

if not os.path.exists(style_path):
    print(f"Ошибка: файл стилей не найден по пути {style_path}")
if not os.path.exists(icon_path):
    print(f"Ошибка: иконка не найдена по пути {icon_path}")

if os.path.exists(style_path):
    with open(style_path, "r") as f:
        style_sheet = f.read()
    icon_path_escaped = icon_path.replace("\\", "/")
    style_sheet = style_sheet.replace("url(iconka.png)", f"url({icon_path_escaped})")
    app.setStyleSheet(style_sheet)
    print("Стили успешно загружены и модифицированы")
else:
    print("Файл стилей не найден, приложение запущено без стилей")



stacked_widget = QStackedWidget()
window.setCentralWidget(stacked_widget)

main_page = QWidget()
main_layout = QGridLayout(main_page)

button1 = QPushButton("загрузить файл")
button2 = QPushButton("создать матрицу")
output = QLabel("Выберите формат загрузки данных")
output.setAlignment(Qt.AlignCenter)
table = QTableWidget()
table.hide()

back_to_main_button = QPushButton("Назад")
back_to_main_button.hide()

matrix_container = QWidget()
matrix_container_layout = QHBoxLayout(matrix_container)
matrix_container_layout.setContentsMargins(0, 0, 0, 0)

run_button = QPushButton("Запустить")
run_button.hide()

main_layout.addWidget(button1, 0, 0)
main_layout.addWidget(button2, 0, 1)
main_layout.addWidget(back_to_main_button, 0, 0, 1, 2, alignment=Qt.AlignLeft | Qt.AlignTop)
main_layout.addWidget(output, 1, 0, 1, 2)

matrix_buttons = QWidget()
matrix_buttons_layout = QVBoxLayout(matrix_buttons)
matrix_buttons_layout.setAlignment(Qt.AlignTop)
matrix_buttons_layout.setSpacing(30)
matrix_buttons.hide()
matrix_container_layout.addWidget(matrix_buttons, stretch=0)
matrix_container_layout.addWidget(table, stretch=1)
main_layout.addWidget(matrix_container, 2, 0, 1, 2)

graph_page = QWidget()
main_graph_layout = QVBoxLayout(graph_page)

top_bar = QHBoxLayout()
back_button = QPushButton("Назад")
refresh_button = QPushButton("Перерисовать граф")
t_norm_combo = QComboBox()
t_norm_combo.addItems(['min', 'product', 'hamacher', 'einstein', 'nilpotent', 'lukasiewicz'])

s_norm_combo = QComboBox()
s_norm_combo.addItems(['max', 'bounded_sum', 'algebraic_sum', 'hamacher', 'einstein', 'nilpotent_max', 'lukasiewicz'])

top_bar.addWidget(back_button)
top_bar.addWidget(refresh_button)
top_bar.addStretch()
top_bar.addWidget(QLabel("Выберите T-норму:"))
top_bar.addWidget(t_norm_combo)
top_bar.addWidget(QLabel("Выберите S-норму:"))
top_bar.addWidget(s_norm_combo)

main_graph_layout.addLayout(top_bar)

content_layout = QHBoxLayout()
content_layout.setContentsMargins(0, 0, 0, 0)
content_layout.setSpacing(15)

side_panel = QWidget()
side_panel.setFixedWidth(280)
side_panel_layout = QVBoxLayout(side_panel)
side_panel_layout.setAlignment(Qt.AlignTop)
side_panel_layout.setSpacing(10)

q_label = QLabel("Внешнее воздействие:")
q_input = QLineEdit()
q_input.setPlaceholderText("0,0,0.5,0...")

o_label = QLabel("Управляющие воздействие:")
o_input = QLineEdit()
o_input.setPlaceholderText("0,0.2,0...")

steps_label = QLabel("Количество шагов:")
steps_spin = QSpinBox()
steps_spin.setRange(1, 50)
steps_spin.setValue(10)

impulse_btn = QPushButton("Импульс")
save_button = QPushButton("Сохранить")
alg_hebba_button = QPushButton("Алгроритм Хебба")

side_panel_layout.addWidget(q_label)
side_panel_layout.addWidget(q_input)
side_panel_layout.addWidget(o_label)
side_panel_layout.addWidget(o_input)
side_panel_layout.addWidget(steps_label)
side_panel_layout.addWidget(steps_spin)
side_panel_layout.addWidget(impulse_btn)
side_panel_layout.addWidget(alg_hebba_button)
side_panel_layout.addWidget(save_button)
side_panel_layout.addStretch()

graph_container = QWidget()
graph_container_layout = QHBoxLayout(graph_container)
graph_stack = QStackedWidget()

placeholder_label = QLabel("Загрузите данные для отображения")
placeholder_label.setAlignment(Qt.AlignCenter)
graph_stack.addWidget(placeholder_label)

figure = Figure()
figure_canvas = FigureCanvas(figure)
graph_stack.addWidget(figure_canvas)

graph_container_layout.addWidget(graph_stack)

content_layout.addWidget(side_panel)
content_layout.addWidget(graph_container, stretch=1)

main_graph_layout.addLayout(content_layout)

stacked_widget.addWidget(main_page)
stacked_widget.addWidget(graph_page)


matrix_type_combo = QComboBox()
matrix_type_combo.addItems([
    'Исходная матрица',
    'Консонанс',
    'Диссонанс',
    'Воздействие',
    'Влияние на систему',
    'Влияние системы',
    'Совместное влияние'
])

threshold_input = QLineEdit()
threshold_input.setPlaceholderText("Порог среза")
threshold_input.setValidator(QDoubleValidator(0.0, 1.0, 3))

top_bar.addWidget(QLabel("Тип матрицы:"))
top_bar.addWidget(matrix_type_combo)
top_bar.addWidget(QLabel("Порог:"))
top_bar.addWidget(threshold_input)


def refresh_graph():
    if graph_stack.count() > 2:
        old_widget = graph_stack.widget(2)
        graph_stack.removeWidget(old_widget)
        old_widget.deleteLater()

    matrix_type_map = {
        'Исходная матрица': 'original',
        'Консонанс': 'consonance',
        'Диссонанс': 'dissonance',
        'Воздействие': 'impact',
        'Влияние на систему': 'system_influence_i',
        'Влияние системы': 'system_influence_j',
        'Совместное влияние': 'joint_positive'
    }

    try:
        threshold_text = threshold_input.text().replace(',', '.').strip()
        threshold = float(threshold_text) if threshold_text else 0.0
    except ValueError:
        threshold = 0.0

    web_view = setup_fcm(
        window,
        column,
        matrix,
        t_norm_combo.currentText(),
        s_norm_combo.currentText(),
        matrix_type_map[matrix_type_combo.currentText()],
        threshold
    )

    if web_view:
        graph_stack.addWidget(web_view)
        graph_stack.setCurrentIndex(2)


def on_open_file():
    global matrix, column
    matrix_buttons.hide()
    table.hide()
    output.setText("Выберите формат загрузки данных")
    run_button.hide()

    files, _ = QFileDialog.getOpenFileName(
        window, "Выберите файл", "",
        "Текстовые файлы (*.txt);;CSV файлы (*.csv);;Все файлы (*.*);;Excel файлы (*.xlsx)")

    if files:
        output.setText("Расположение: " + files)
        file_extention = os.path.splitext(files)[1].lower()

        column.clear()
        matrix.clear()
        button1.hide()
        button2.hide()
        back_to_main_button.show()

        if file_extention == '.txt':
            try:
                with open(files, 'r', encoding='utf-8') as file:
                    lines = file.readline()
                    column = lines.split()
                    for line in file:
                        matrix_str = list(map(int, line.strip().split()))
                        matrix.append(matrix_str)
            except Exception as s:
                QMessageBox.critical(window, "Ошибка", f"Не удалось прочитать файл: {str(s)}")

        elif file_extention == '.csv':
            try:
                with open(files, 'r', encoding='utf-8') as f:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)

                df = pd.read_csv(files, header=None, encoding='utf-8', sep=dialect.delimiter)
                column = df.iloc[0].tolist()

                matrix = []
                for row in df.iloc[1:].values:
                    processed_row = []
                    for value in row:
                        try:
                            processed_row.append(float(value))
                        except (ValueError, TypeError):
                            processed_row.append(0)
                    matrix.append(processed_row)

            except csv.Error:
                try:
                    df = pd.read_csv(files, header=None, encoding='utf-8')
                    column = df.iloc[0].tolist()
                except Exception as e:
                    QMessageBox.critical(window, "Ошибка", f"Ошибка CSV: {e}")
            except Exception as e:
                QMessageBox.critical(window, "Ошибка", f"Не удалось прочитать CSV: {e}")

        elif file_extention == '.xlsx':
            try:
                df = pd.read_excel(files, header=None, engine='openpyxl')
                column = df.iloc[0].tolist()

                matrix = []
                for row in df.iloc[1:].values:
                    processed_row = []
                    for value in row:
                        try:
                            processed_row.append(float(value))
                        except (ValueError, TypeError):
                            processed_row.append(0)
                    matrix.append(processed_row)

            except Exception as a:
                QMessageBox.critical(window, "Ошибка", f"Не удалось прочитать файл: {a}")

        main_layout.addWidget(run_button, 4, 0, 1, 2, alignment=Qt.AlignHCenter | Qt.AlignHCenter)
        run_button.show()


def open_fcm_file():
    if graph_stack.count() > 2:
        old_widget = graph_stack.widget(2)
        graph_stack.removeWidget(old_widget)
        old_widget.deleteLater()

    t_norm = t_norm_combo.currentText()
    s_norm = s_norm_combo.currentText()
    web_view = setup_fcm(window, column, matrix, t_norm, s_norm)

    if web_view:
        graph_stack.addWidget(web_view)
        graph_stack.setCurrentIndex(2)
        stacked_widget.setCurrentIndex(1)


def on_create_matrix():
    button1.hide()
    button2.hide()
    back_to_main_button.show()
    global matrix, column, run_button
    output.setText("Заполните матрицу")
    matrix.clear()
    column.clear()

    while matrix_buttons_layout.count():
        child = matrix_buttons_layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()

    run_button = QPushButton("Запустить")
    run_button.clicked.connect(open_fcm_file)

    add_col_button = QPushButton("Добавьте название колонки")
    add_col_button.clicked.connect(add_column)
    matrix_buttons_layout.addWidget(add_col_button)

    add_row_button = QPushButton("Добавьте строки")
    add_row_button.clicked.connect(add_row)
    matrix_buttons_layout.addWidget(add_row_button)

    delet_row_button = QPushButton("Удалить строку")
    delet_row_button.clicked.connect(delete_row)
    matrix_buttons_layout.addWidget(delet_row_button)

    delet_col_button = QPushButton("Удалить столбец")
    delet_col_button.clicked.connect(delete_selected_column)
    matrix_buttons_layout.addWidget(delet_col_button)

    matrix_buttons_layout.addStretch()
    matrix_buttons_layout.addWidget(run_button, alignment=Qt.AlignBottom)

    matrix_buttons.show()
    run_button.show()
    update_matrix_display()


def add_column():
    global matrix, column
    column_name, ok = QInputDialog.getText(window, "Добавить столбец", "Введите название столбца:")
    if ok and column_name:
        if column_name not in column:
            column.append(column_name)
            if len(matrix) > 0:
                for i in range(len(matrix)):
                    matrix[i].append(0)
            update_matrix_display()
        else:
            QMessageBox.critical(window, "Ошибка", "Введите корректное название столбца")


def delete_row():
    global matrix
    selected_rows = table.selectionModel().selectedRows()
    if not selected_rows:
        QMessageBox.warning(window, "Ошибка", "Строка не выбрана! Выделите строку для удаления.")
        return

    reply = QMessageBox.question(window, "Подтверждение", "Вы действительно хотите удалить эту строку?",
                                 QMessageBox.Yes | QMessageBox.No)
    if reply == QMessageBox.Yes:
        try:
            row_ind = selected_rows[0].row()
            del matrix[row_ind]
            update_matrix_display()
        except Exception as a:
            QMessageBox.critical(window, "Ошибка", f"Не удалось удалить строку: {str(a)}")


def add_row():
    global matrix
    if len(column) == 0:
        QMessageBox.critical(window, "Ошибка", "Сначала добавьте столбцы")
        return

    row_count, ok = QInputDialog.getInt(window, "Добавить строки", "Сколько строк добавить?",
                                        min=1, max=100)
    if ok and row_count:
        for _ in range(row_count):
            matrix.append([0] * len(column))
        update_matrix_display()
        print(matrix)


def update_matrix_display():
    table.show()
    table.clear()

    try:
        table.horizontalHeader().sectionDoubleClicked.disconnect()
        table.cellDoubleClicked.disconnect()
    except:
        pass

    table.setRowCount(len(matrix))
    table.setColumnCount(len(column))
    table.setHorizontalHeaderLabels(column)
    table.setVerticalHeaderLabels(column)

    table.horizontalHeader().setSectionsClickable(True)

    table.horizontalHeader().sectionDoubleClicked.connect(delete_column_by_header)
    table.cellDoubleClicked.connect(edit_cell)

    for i, row_data in enumerate(matrix):
        for j, value in enumerate(row_data):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, j, item)


def delete_column_by_header(logicalIndex):
    global matrix
    selected_rows = table.selectionModel().selectedColumns()
    if not selected_rows:
        QMessageBox.warning(window, "Ошибка", "Строка не выбрана! Выделите строку для удаления.")
        return

    reply = QMessageBox.question(window, "Подтверждение",
                                 "Вы действительно хотите удалить этот столбец?",
                                 QMessageBox.Yes | QMessageBox.No)

    if reply == QMessageBox.Yes:
        try:
            del column[logicalIndex]

            for row in matrix:
                del row[logicalIndex]

            if len(column) == 0:
                matrix.clear()

            update_matrix_display()

        except Exception as e:
            QMessageBox.critical(window, "Ошибка", f"Не удалось удалить столбец: {str(e)}")


def delete_selected_column():
    selected_columns = table.selectionModel().selectedColumns()

    if not selected_columns:
        QMessageBox.warning(window, "Ошибка", "Строка не выбрана! Выделите строку для удаления.")
        return

    columns_to_delete = {col.column() for col in selected_columns}

    reply = QMessageBox.question(
        window,
        "Подтверждение",
        f"Удалить {len(columns_to_delete)} столбцов?",
        QMessageBox.Yes | QMessageBox.No
    )

    if reply == QMessageBox.Yes:
        try:
            for col in sorted(columns_to_delete, reverse=True):
                del column[col]
                for row in matrix:
                    del row[col]

            if len(column) == 0:
                matrix.clear()

            update_matrix_display()
        except Exception as e:
            QMessageBox.critical(window, "Ошибка", f"Ошибка удаления: {str(e)}")

def edit_cell(row, column):
    if hasattr(window, '_edit_dialog_open') and window._edit_dialog_open:
        return

    current_value = table.item(row, column).text()

    dialog = QDialog(window)
    dialog.setWindowTitle("Редактирование")
    layout = QVBoxLayout(dialog)

    entry = QLineEdit(dialog)
    entry.setText(current_value)
    layout.addWidget(entry)

    save_btn = QPushButton("Сохранить", dialog)
    save_btn.clicked.connect(lambda: save_cell_value(row, column, entry.text(), dialog))
    layout.addWidget(save_btn)

    window._edit_dialog_open = True
    dialog.finished.connect(lambda: setattr(window, '_edit_dialog_open', False))

    dialog.exec_()

    table.clearFocus()
    table.setCurrentCell(-1, -1)


def save_cell_value(row, col, value, dialog):
    try:
        new_value = float(value)
        matrix[row][col] = new_value
        item = QTableWidgetItem(str(new_value))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, col, item)
        dialog.close()
    except ValueError:
        QMessageBox.critical(dialog, "Ошибка", "Вводите только числа!")


def back_to_main():
    stacked_widget.setCurrentIndex(0)
    graph_stack.setCurrentIndex(0)


def run_impulse():
    try:
        q_text = q_input.text().strip()
        o_text = o_input.text().strip()

        if not q_text and not o_text:
            QMessageBox.warning(window, "Ошибка", "Заполните хотя бы одно из полей воздействия!")
            return

        q = np.array([float(x.strip()) for x in q_text.split(',')]) if q_text else np.zeros(len(matrix[0]))
        o = np.array([float(x.strip()) for x in o_text.split(',')]) if o_text else np.zeros(len(matrix[0]))

        if len(q) != len(matrix[0]) or len(o) != len(matrix[0]):
            raise ValueError(f"Длина векторов должна быть {len(matrix[0])}")

        steps = steps_spin.value()

        history = run_impulse_simulation(
            matrix=matrix,
            steps=steps,
            q=q,
            o=o,
            t_norm=t_norm_combo.currentText(),
            s_norm=s_norm_combo.currentText()
        )

        figure.clear()
        ax = figure.add_subplot(111)
        for i in range(history.shape[1]):
            ax.plot(history[:, i], label=column[i])
        ax.legend()
        ax.set_xlabel("Шаги")
        ax.set_ylabel("Значение")
        figure_canvas.draw()
        graph_stack.setCurrentIndex(1)

    except ValueError as e:
        QMessageBox.critical(window, "Ошибка ввода", str(e))
    except RuntimeError as e:
        QMessageBox.critical(window, "Ошибка моделирования", str(e))
    except Exception as e:
        QMessageBox.critical(window, "Неизвестная ошибка", str(e))


def save_data():
    data_matrix, impulse_history = get_global_data()

    if data_matrix is None and impulse_history is None:
        QMessageBox.warning(window, "Ошибка", "Нет данных для сохранения")
        return

    if data_matrix is not None:
        headers = column
        data = data_matrix
    else:
        headers = column
        data = impulse_history

    file_path, selected_filter = QFileDialog.getSaveFileName(
        window,
        "Сохранить данные FCM",
        "FCM_result",
        "CSV файлы (*.csv);;Текстовые файлы (*.txt);;Excel файлы (*.xlsx)"
    )

    if not file_path:
        return

    if selected_filter == "CSV файлы (*.csv)":
        file_format = "csv"
    elif selected_filter == "Текстовые файлы (*.txt)":
        file_format = "txt"
    elif selected_filter == "Excel файлы (*.xlsx)":
        file_format = "xlsx"
    else:
        if file_path.lower().endswith('.csv'):
            file_format = "csv"
        elif file_path.lower().endswith('.txt'):
            file_format = "txt"
        elif file_path.lower().endswith('.xlsx'):
            file_format = "xlsx"
        else:
            file_path += ".csv"
            file_format = "csv"

    try:
        if file_format == "csv" or file_format == "txt":
            delimiter = ',' if file_format == "csv" else '\t'

            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=delimiter)

                if data_matrix is not None:
                    data = data_matrix

                    if data.ndim == 1:
                        writer.writerow(["Concept", "Value"])
                        for i, name in enumerate(column):
                            writer.writerow([name, data[i]])
                    else:
                        writer.writerow([""] + column)
                        for i, name in enumerate(column):
                            row = [name] + data[i].tolist()
                            writer.writerow(row)

                elif impulse_history is not None:
                    writer.writerow(["Step"] + column)
                    for step in range(impulse_history.shape[0]):
                        row = [step] + impulse_history[step].tolist()
                        writer.writerow(row)

        elif file_format == "xlsx":
            try:
                import pandas as pd
            except ImportError:
                QMessageBox.critical(
                    window,
                    "Ошибка",
                    "Для сохранения в Excel требуется установить библиотеку pandas.\n"
                    "Установите ее командой: pip install pandas openpyxl"
                )
                return

            if data_matrix is not None:
                data = data_matrix

                if data.ndim == 1:
                    df = pd.DataFrame({
                        "Concept": column,
                        "Value": data
                    })
                else:
                    df = pd.DataFrame(data, columns=column)
                    df.insert(0, "Concept", column)

            elif impulse_history is not None:
                df = pd.DataFrame(
                    impulse_history,
                    columns=column
                )
                df.insert(0, "Step", range(len(impulse_history)))

            df.to_excel(file_path, index=False)

        QMessageBox.information(
            window,
            "Успех",
            f"Данные успешно сохранены в:\n{file_path}"
        )

    except Exception as e:
        QMessageBox.critical(
            window,
            "Ошибка сохранения",
            f"Произошла ошибка при сохранении файла:\n{str(e)}"
        )
def read_initial_values(file_path):
    import os
    import pandas as pd

    file_extension = os.path.splitext(file_path)[1].lower()
    data = []

    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.replace(',', ' ').split()
                for value in values:
                    try:
                        num = float(value)
                        data.append(num)
                    except ValueError:
                        continue

    elif file_extension in ('.xlsx', '.csv'):
        try:
            if file_extension == '.xlsx':
                df = pd.read_excel(file_path, header=None, engine='openpyxl')
            else:
                df = pd.read_csv(file_path, header=None, sep=None, engine='python')

            for value in df.values.flatten():
                try:
                    num = float(value)
                    data.append(num)
                except (ValueError, TypeError):
                    continue

        except Exception as e:
            raise ValueError(f"Ошибка чтения файла: {str(e)}")

    else:
        raise ValueError("Неподдерживаемый формат файла")

    return data

def on_alg_hebba_button_clicked():
    file_path, _ = QFileDialog.getOpenFileName(
        window,
        "Выберите файл",
        "",
        "CSV файлы (*.csv);;Текстовые файлы (*.txt);;Excel файлы (*.xlsx)"
    )

    if not file_path:
        return

    if os.path.getsize(file_path) == 0:
        QMessageBox.critical(window, "Ошибка", "Выбранный файл пуст!")
        return

    try:
        initial_values = read_initial_values(file_path)

        matrix_type_map = {
            'Исходная матрица': 'original',
            'Консонанс': 'consonance',
            'Диссонанс': 'dissonance',
            'Воздействие': 'impact',
            'Влияние на систему': 'system_influence_i',
            'Влияние системы': 'system_influence_j',
            'Совместное влияние': 'joint_positive'
        }

        try:
            if graph_stack.count() > 2:
                old_widget = graph_stack.widget(2)
                graph_stack.removeWidget(old_widget)
                old_widget.deleteLater()

            try:
                threshold_text = threshold_input.text().replace(',', '.').strip()
                threshold = float(threshold_text) if threshold_text else 0.0
            except ValueError:
                threshold = 0.0

            web_view = alg_hebba(
                window,
                column,
                matrix,
                initial_values,
                matrix_type_map[matrix_type_combo.currentText()],
                threshold
            )

            if web_view:
                graph_stack.addWidget(web_view)
                graph_stack.setCurrentIndex(2)

            QMessageBox.information(
                window,
                "Успех",
                "Обучение алгоритмом Хебба завершено успешно!"
            )

        except Exception as e:
            QMessageBox.critical(
                window,
                "Ошибка выполнения алгоритма",
                f"Произошла ошибка во время обучения:\n{str(e)}"
            )

    except Exception as e:
        QMessageBox.critical(
            window,
            "Ошибка обработки файла",
            f"Ошибка при чтении файла:\n{str(e)}"
        )
        return

def return_to_main():
    button1.show()
    button2.show()
    run_button.hide()
    back_to_main_button.hide()

    matrix_buttons.hide()
    table.hide()
    output.setText("Выберите формат загрузки данных")

    global matrix, column
    matrix.clear()
    column.clear()

    if run_button.parent():
        run_button.parent().layout().removeWidget(run_button)
button1.clicked.connect(on_open_file)
button2.clicked.connect(on_create_matrix)
run_button.clicked.connect(open_fcm_file)
back_to_main_button.clicked.connect(return_to_main)
impulse_btn.clicked.connect(run_impulse)
save_button.clicked.connect(save_data)
alg_hebba_button.clicked.connect(on_alg_hebba_button_clicked)
refresh_button.clicked.connect(refresh_graph)
back_button.clicked.connect(back_to_main)
window.show()
sys.exit(app.exec_())