import numpy as np
import pyvis
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from pyvis.network import Network
from PyQt5.QtWidgets import QMessageBox
import tempfile
import os

global_data_matrix = None
global_impulse_history = None

def build_double_matrix(matrix):
    n = len(matrix)
    R = np.zeros((2 * n, 2 * n))

    for i in range(n):
        for j in range(n):
            wij = matrix[i][j]
            if wij > 0:
                R[2 * i, 2 * j] = wij
                R[2 * i + 1, 2 * j + 1] = wij
            elif wij < 0:
                R[2 * i, 2 * j + 1] = -wij
                R[2 * i + 1, 2 * j] = -wij
    return R

snorm_functions = {
    'max': lambda a, b: np.maximum(a, b),
    'bounded_sum': lambda a, b: np.minimum(a + b, 1.0),
    'algebraic_sum': lambda a, b: a + b - a*b,
    'hamacher': lambda a, b: (a + b - 2*a*b) / np.where((1 - a*b) == 0, 1e-8, (1 - a*b)),
    'einstein': lambda a, b: (a + b) / (1 + a*b),
    'nilpotent_max': lambda a, b: np.where(a + b < 1, np.maximum(a, b), 1.0),
    'lukasiewicz': lambda a, b: np.minimum(a + b, 1.0)
}

Tnorm_functions = {
    'min': lambda a, b: np.minimum(a, b),
    'product': lambda a, b: a * b,
    'hamacher': lambda a, b: np.where(
        (a == 0) & (b == 0),
        0,
        (a * b) / np.where(
            (a + b - a*b) == 0,
            1e-8,
            (a + b - a*b)
        )
    ),
    'einstein': lambda a, b: np.where(
        (a == 0) & (b == 0),
        0,
        (a * b) / np.where(
            (2 - (a + b - a*b)) == 0,
            1e-8,
            (2 - (a + b - a*b))
        )
    ),
    'nilpotent': lambda a, b: np.where(a + b >= 1, np.minimum(a, b), 0),
    'lukasiewicz': lambda a, b: np.maximum(a + b - 1, 0)
}


def compose(A, B, t_norm, s_norm):
    n = A.shape[0]
    m = A.shape[1]
    p = B.shape[1]
    result = np.zeros((n, p))

    for i in range(n):
        for j in range(p):
            accum = None
            for k in range(m):
                temp = t_norm(A[i, k], B[k, j])
                if accum is None:
                    accum = temp
                else:
                    accum = s_norm(accum, temp)
            result[i, j] = accum if accum is not None else 0.0
    return result


def transitive_closure(R, t_norm='min', s_norm='max', max_iter=100):
    s_norm_func = snorm_functions[s_norm]
    t_norm_func = Tnorm_functions[t_norm]

    R_current = R.copy()

    for _ in range(max_iter):
        R_comp = compose(R_current, R_current, t_norm_func, s_norm_func)
        R_next = s_norm_func(R_current, R_comp)

        if np.array_equal(R_next, R_current):
            break
        R_current = R_next

    return R_current


def get_v_matrix(R_trans, n):
    V = np.zeros((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            pos = max(R_trans[2 * i, 2 * j], R_trans[2 * i + 1, 2 * j + 1])
            neg = -max(R_trans[2 * i, 2 * j + 1], R_trans[2 * i + 1, 2 * j])
            V[i, j] = [round(pos, 3), round(neg, 3)]
    return V



def get_consonance(V, n):

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pos, neg = V[i][j]
            denominator = abs(pos) + abs(neg)
            if denominator == 0:
                C[i, j] = 0.0
            else:
                C[i, j] = round(abs(pos + neg) / denominator, 3)
    return C

def get_dissonance(V, n):
    C = get_consonance(V, n)
    return np.round(1 - C, 3)

def get_impact(V, n):
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pos, neg = V[i][j]
            if pos != -neg:
                sign = np.sign(pos + neg)
                max_val = max(abs(pos), abs(neg))
                P[i, j] = round(sign * max_val, 3)
            else:
                P[i, j] = 0.0
    return P

def get_system_influence_from_i(P):
    return np.round(np.mean(P, axis=1), 3)

def get_system_influence_to_j(P):
    return np.round(np.mean(P, axis=0), 3)

def get_joint_positive_impact(V, n):
    P_joint = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v_ij = V[i][j][0]
            v_ji = V[j][i][0]
            P_joint[i, j] = round(v_ij * v_ji, 3)
    return P_joint




#Импульс
def impulse_update(W, A_current, q, o, T_norm, S_norm):

    influence = np.zeros_like(A_current)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            influence[i] += T_norm(W[i][j], A_current[j])

    combined_input = q + o + influence
    return S_norm(A_current, combined_input)


def impulse(matrix, num_steps, q, o, T_norm, S_norm, A_initial=None):
    K = matrix.shape[0]
    history = np.zeros((num_steps + 1, K))

    history[0] = A_initial if A_initial is not None else np.zeros(K)

    for t in range(num_steps):
        history[t + 1] = impulse_update(matrix, history[t], q, o, T_norm, S_norm)

    return history




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fcm_nhl_train(initial_weights, initial_values, learning_rate=0.001,
                  decay_rate=0.98, max_iter=1000, tol=0.001):

    W = initial_weights.copy()
    A_prev = initial_values.copy()
    num_concepts = len(initial_values)
    original_nonzero = (initial_weights != 0)

    doc_index = num_concepts - 1

    for iteration in range(max_iter):

        A_new = np.zeros(num_concepts)
        for i in range(num_concepts):

            input_val = A_prev[i]
            for j in range(num_concepts):
                if i != j:
                    input_val += A_prev[j] * W[j, i]
            A_new[i] = sigmoid(input_val)

        for i in range(num_concepts):
            for j in range(num_concepts):
                if original_nonzero[j, i]:
                    delta_w = learning_rate * A_prev[i] * (A_prev[j] - W[j, i] * A_prev[i])
                    W[j, i] = decay_rate * W[j, i] + delta_w

        doc_value = A_new[doc_index]
        doc_change = np.abs(doc_value - A_prev[doc_index])

        A_prev = A_new

    return W, A_prev












def setup_fcm(parent_window, column_name, matrix, t_norm='min', s_norm='max' ,matrix_type='original', threshold=0.0):
    global global_data_matrix
    try:

        import sys
        import os
        import pyvis

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            template_dir = os.path.join(base_path, 'pyvis', 'templates')
        else:
            template_dir = os.path.join(os.path.dirname(pyvis.__file__), 'templates')

        os.environ['PYVIS_TEMPLATE_DIR'] = template_dir

        if not column_name or not matrix:
            QMessageBox.critical(parent_window, "Ошибка", "Получены пустые данные")
            return None

        reset_global_data()

        n = len(matrix)

        R = build_double_matrix(matrix)

        R_trans = transitive_closure(R, t_norm, s_norm)

        V = get_v_matrix(R_trans, n)

        if matrix_type == 'consonance':
            data = get_consonance(V, n)
        elif matrix_type == 'dissonance':
            data = get_dissonance(V, n)
        elif matrix_type == 'impact':
            data = get_impact(V, n)
        elif matrix_type == 'system_influence_i':
            data = get_system_influence_from_i(get_impact(V, n))
        elif matrix_type == 'system_influence_j':
            data = get_system_influence_to_j(get_impact(V, n))
        elif matrix_type == 'joint_positive':
            data = get_joint_positive_impact(V, n)
        else:
            data = matrix

            data = np.array(data, dtype=float)

        global_data_matrix = data

        net = Network(height="750px", width="100%", directed=True, notebook=False)

        net.set_options("""
        {
          "edges": {
            "font": {
              "size": 14,
              "strokeWidth": 0
            },
            "smooth": {
              "type": "continuous"
            }
          },
        
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.1,
              "springLength": 400
            }
          }
        }
        """)

        for i, name in enumerate(column_name):
            net.add_node(i, label=name, size=20, font={'size': 30})

        if data.ndim == 1:
            max_val = np.max(np.abs(data)) if data.size > 0 else 1.0
            for i in range(n):
                value = data[i]
                if abs(value) >= threshold and not np.isclose(value, 0.0, atol=1e-6):
                    net.add_edge(i, i,
                                 value=abs(value),
                                 title=f"Влияние: {value:.3f}",
                                 label=f"{value:.2f}",
                                 width=1 + 5 * abs(value) / max_val,
                                 color='red' if value < 0 else 'green')
        else:
            max_val = np.max(np.abs(data)) if data.size > 0 else 1.0
            for i in range(n):
                for j in range(n):
                    value = data[i][j]
                    if abs(value) >= threshold and not np.isclose(value, 0.0, atol=1e-6):
                        net.add_edge(i, j,
                                     value=abs(value),
                                     title=f"{value:.3f}",
                                     label=f"{value:.2f}",
                                     width=1 + 5 * abs(value) / max_val,
                                     color='red' if value < 0 else 'green')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        temp_path = temp_file.name
        net.save_graph(temp_path)
        temp_file.close()

        web_view = QWebEngineView()
        web_view.load(QUrl.fromLocalFile(os.path.abspath(temp_path)))

        def cleanup():
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Ошибка удаления файла: {e}")

        web_view.loadFinished.connect(cleanup)
        return web_view

    except Exception as e:
        QMessageBox.critical(parent_window, "Ошибка", f"Не удалось построить граф: {str(e)}")
        return None








def run_impulse_simulation(matrix, steps, q, o, t_norm, s_norm):

    global global_impulse_history
    try:
        reset_global_data()
        W = np.array(matrix, dtype=float)

        T_func = Tnorm_functions.get(t_norm)
        S_func = snorm_functions.get(s_norm)

        if not T_func or not S_func:
            raise ValueError("Некорректно выбраны T-норма или S-норма")

        K = W.shape[0]
        history = np.zeros((steps + 1, K))

        history[0] = np.zeros(K)

        for t in range(steps):
            history[t + 1] = impulse_update(
                W,
                history[t],
                q,
                o,
                T_func,
                S_func
            )

        global_impulse_history = history
        return history

    except Exception as e:
        raise RuntimeError(f"Ошибка при выполнении импульса: {str(e)}")



def reset_global_data():
    global global_data_matrix, global_impulse_history
    global_data_matrix = None
    global_impulse_history = None


def get_global_data():

    global global_data_matrix, global_impulse_history
    return global_data_matrix, global_impulse_history


def alg_hebba(parent_window, column_names, initial_weights, initial_values, matrix_type='original', threshold=0.0):

    try:

        import sys
        import os
        import pyvis

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            template_dir = os.path.join(base_path, 'pyvis', 'templates')
        else:
            template_dir = os.path.join(os.path.dirname(pyvis.__file__), 'templates')

        os.environ['PYVIS_TEMPLATE_DIR'] = template_dir

        if not isinstance(initial_weights, np.ndarray):
            initial_weights = np.array(initial_weights)

        updated_weights, final_values = fcm_nhl_train(
            initial_weights,
            initial_values
        )

        if not isinstance(updated_weights, np.ndarray):
            updated_weights = np.array(updated_weights)

        n = len(updated_weights)

        R = build_double_matrix(updated_weights)

        V = get_v_matrix(R, n)

        if matrix_type == 'consonance':
            data = get_consonance(V, n)
        elif matrix_type == 'dissonance':
            data = get_dissonance(V, n)
        elif matrix_type == 'impact':
            data = get_impact(V, n)
        elif matrix_type == 'system_influence_i':
            data = get_system_influence_from_i(get_impact(V, n))
        elif matrix_type == 'system_influence_j':
            data = get_system_influence_to_j(get_impact(V, n))
        elif matrix_type == 'joint_positive':
            data = get_joint_positive_impact(V, n)
        else:
            data = updated_weights

        data = np.array(data, dtype=float)

        net = Network(height="750px", width="100%", directed=True, notebook=False)
        net.set_options("""
        {
          "edges": {
            "font": {
              "size": 14,
              "strokeWidth": 0
            },
            "smooth": {
              "type": "continuous"
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.1,
              "springLength": 400
            }
          }
        }
        """)

        for i, name in enumerate(column_names):
            net.add_node(i, label=name, size=20, font={'size': 30})

        if data.ndim == 1:
            max_val = np.max(np.abs(data)) if data.size > 0 else 1.0
            for i in range(n):
                value = data[i]
                if abs(value) >= threshold and not np.isclose(value, 0.0, atol=1e-6):
                    net.add_edge(i, i,
                                 value=abs(value),
                                 title=f"Влияние: {value:.3f}",
                                 label=f"{value:.2f}",
                                 width=1 + 5 * abs(value) / max_val,
                                 color='red' if value < 0 else 'green')
        else:

            max_val = np.max(np.abs(data)) if data.size > 0 else 1.0
            for i in range(n):
                for j in range(n):
                    value = data[i][j]
                    if abs(value) >= threshold and not np.isclose(value, 0.0, atol=1e-6):
                        net.add_edge(i, j,
                                     value=abs(value),
                                     title=f"{value:.3f}",
                                     label=f"{value:.2f}",
                                     width=1 + 5 * abs(value) / max_val,
                                     color='red' if value < 0 else 'green')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        temp_path = temp_file.name
        net.save_graph(temp_path)
        temp_file.close()

        web_view = QWebEngineView()
        web_view.load(QUrl.fromLocalFile(os.path.abspath(temp_path)))

        def cleanup():
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Ошибка удаления файла: {e}")

        web_view.loadFinished.connect(cleanup)
        return web_view

    except Exception as e:
        QMessageBox.critical(
            parent_window,
            "Ошибка",
            f"Не удалось выполнить алгоритм Хебба: {str(e)}"
        )
        return None

